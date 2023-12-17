import torch
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
import shutil
from datetime import datetime


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        # log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        # if not args.debug:
        # self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (week_data, month_data, recent_data, target) in enumerate(val_dataloader):
                month_data = month_data.permute(0, 3, 1, 2)
                week_data = week_data.permute(0, 3, 1, 2)
                label = target.permute(0, 3, 1, 2)
                output = self.model(week_data, month_data, recent_data)

                loss = self.loss(output.cuda(), label)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (week_data, month_data, recent_data, target) in enumerate(self.train_loader):
            month_data = month_data.permute(0, 3, 1, 2)
            week_data = week_data.permute(0, 3, 1, 2)
            label = target.permute(0, 3, 1, 2)
            self.optimizer.zero_grad()

            # teacher_forcing for RNN encoder-decoder model
            # if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.

            output = self.model(week_data, month_data, recent_data)

            loss = self.loss(output.cuda(), label)
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            # log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss / self.train_per_epoch
        self.logger.info(
            '**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss,
                                                                                       teacher_forcing_ratio))

        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        params_path = "experiments/" + self.args.data_path + datetime.now().strftime('%Y%m%d%H%M%S')
        if (self.args.start_epoch == 0) and (not os.path.exists(params_path)):
            os.makedirs(params_path)
            print('create params directory %s' % params_path)
        elif (self.args.start_epoch == 0) and (os.path.exists(params_path)):
            shutil.rmtree(params_path)
            os.makedirs(params_path)
            print('delete the old one and create params directory %s' % (params_path))
        elif (self.args.start_epoch > 0) and (os.path.exists(params_path)):
            print('train from params directory %s' % params_path)
        else:
            raise SystemExit('Wrong type of model!')
        if self.args.start_epoch > 0:
            params_filename = os.path.join(params_path, 'epoch_%s.pth' % self.args.start_epoch)

            self.model.load_state_dict(torch.load(params_filename))

            print('start epoch:', self.args.start_epoch)

            print('load weight from: ', params_filename)
        train_time = []
        val_time = []
        for epoch in range(self.args.start_epoch, self.args.epochs + 1):
            params_filename = os.path.join(params_path, 'epoch_%s.pth' % epoch)
            # epoch_time = time.time()
            train_start = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            train_end = time.time()
            train_time.append(train_end - train_start)
            # print(time.time()-epoch_time)
            # exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader

            val_strat = time.time()
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            val_end = time.time()
            val_time.append(val_end - val_strat)
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
                torch.save(self.model.state_dict(), params_filename)

            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        self.logger.info("Average Training Time: {:.4f} secs/epoch".format(
            np.mean(train_time)))
        self.logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        # save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        # test
        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader,
                  self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader,
             logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (week_data, month_data, recent_data, target) in enumerate(data_loader):
                month_data = month_data.permute(0, 3, 1, 2)
                week_data = week_data.permute(0, 3, 1, 2)
                label = target.permute(0, 3, 1, 2)
                output = model(week_data, month_data, recent_data)
                y_true.append(label)
                y_pred.append(output)
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        np.save('./{}_{}true.npy'.format(args.dataset, datetime.now().strftime('%Y%m%d%H%M%S')), y_true.cpu().numpy())
        np.save('./{}_{}pred.npy'.format(args.dataset, datetime.now().strftime('%Y%m%d%H%M%S')), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape * 100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
            mae, rmse, mape * 100))

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
