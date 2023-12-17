import os
import numpy as np

def search_data(sequence_length, num_of_batches, label_start_idx,
                num_for_predict, units, points_per_hour, current=0):
    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_batches + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_batches:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_months, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=4):
    month_indices = search_data(data_sequence.shape[0], num_of_months,
                                label_start_idx, num_for_predict,
                                4 * 7 * 24, points_per_hour)
    if not month_indices:
        return None

    week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                               label_start_idx, num_for_predict,
                               7 * 24, points_per_hour)
    if not week_indices:
        return None

    hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                               label_start_idx, num_for_predict,
                               1, points_per_hour, current=1)
    if not hour_indices:
        return None

    week_sample = np.concatenate([data_sequence[i: j]
                                  for i, j in week_indices], axis=0)
    month_sample = np.concatenate([data_sequence[i: j]
                                   for i, j in month_indices], axis=0)
    hour_sample = np.concatenate([data_sequence[i: j]
                                  for i, j in hour_indices], axis=0)
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]
    t = label_start_idx + num_for_predict
    target = target
    return week_sample, month_sample, hour_sample, target

def read_and_generate_dataset(graph_signal_matrix_filename,
                              num_of_weeks, num_of_months,
                              num_of_hours, num_for_predict,
                              points_per_hour=4, times=96, merge=False, save=False):
    data_seq = np.load(graph_signal_matrix_filename)['data'][:, :, 0:1].astype("float32")

    L, N, F = data_seq.shape

    feature_list = [data_seq]
    # numerical time_in_day
    time_ind = [i % times / times for i in range(data_seq.shape[0])]
    time_ind = np.array(time_ind)
    time_in_day = np.tile(time_ind, [1, N, 1]).transpose((2, 1, 0))
    feature_list.append(time_in_day)

    # numerical day_in_week
    day_in_week = [(i // times) % 7 for i in range(data_seq.shape[0])]
    day_in_week = np.array(day_in_week)
    day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((2, 1, 0))
    feature_list.append(day_in_week)

    data_seq = np.concatenate(feature_list, axis=-1)
    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_months,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if not sample:
            continue

        week_sample, month_sample, hour_sample, target = sample
        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(month_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(target[:, :, 0:1], axis=0).transpose((0, 2, 3, 1))
        ))
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    if not merge:
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line1])]
    else:
        print('Merge training set and validation set!')
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line2])]

    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_week, train_month, train_hour, train_target = training_set
    val_week, val_month, val_hour, val_target = validation_set
    test_week, test_month, test_hour, test_target = testing_set
    print("len(all_samples)", len(all_samples))
    print("data_seq.shape", data_seq.shape)
    print('training data: week: {}, month: {}, recent: {}, target: {}'.format(
        train_week.shape, train_month.shape,
        train_hour.shape, train_target.shape))
    print('validation data: week: {}, month: {}, recent: {}, target: {}'.format(
        val_week.shape, val_month.shape, val_hour.shape, val_target.shape))
    print('testing data: week: {}, month: {}, recent: {}, target: {}'.format(
        test_week.shape, test_month.shape, test_hour.shape, test_target.shape))

    (week_stats, train_week_norm,
     val_week_norm, test_week_norm) = normalization(train_week,
                                                    val_week,
                                                    test_week)
    # 由于时间编码 第二维和第三维的时间编码列不能进行归一化处理
    train_week_norm[:, :, 1:2] = train_week[:, :, 1:2]
    train_week_norm[:, :, 2:3] = train_week[:, :, 2:3]

    val_week_norm[:, :, 1:2] = val_week[:, :, 1:2]
    val_week_norm[:, :, 2:3] = val_week[:, :, 2:3]

    test_week_norm[:, :, 1:2] = test_week[:, :, 1:2]
    test_week_norm[:, :, 2:3] = test_week[:, :, 2:3]

    (month_stats, train_month_norm,
     val_month_norm, test_month_norm) = normalization(train_month,
                                                      val_month,
                                                      test_month)

    train_month_norm[:, :, 1:2] = train_month[:, :, 1:2]
    train_month_norm[:, :, 2:3] = train_month[:, :, 2:3]

    val_month_norm[:, :, 1:2] = val_month[:, :, 1:2]
    val_month_norm[:, :, 2:3] = val_month[:, :, 2:3]

    test_month_norm[:, :, 1:3] = test_month[:, :, 1:3]

    (recent_stats, train_recent_norm,
     val_recent_norm, test_recent_norm) = normalization(train_hour,
                                                        val_hour,
                                                        test_hour)
    train_recent_norm[:, :, 1:3] = train_hour[:, :, 1:3]

    val_recent_norm[:, :, 1:3] = val_hour[:, :, 1:3]

    test_recent_norm[:, :, 1:3] = test_hour[:, :, 1:3]

    all_data = {
        'train': {
            'week': train_week_norm,
            'month': train_month_norm,
            'recent': train_recent_norm,
            'target': train_target,
        },
        'val': {
            'week': val_week_norm,
            'month': val_month_norm,
            'recent': val_recent_norm,
            'target': val_target
        },
        'test': {
            'week': test_week_norm,
            'month': test_month_norm,
            'recent': test_recent_norm,
            'target': test_target
        },
        'stats': {
            'week': week_stats,
            'month': month_stats,
            'recent': recent_stats
        }
    }
    if save:
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        filename = os.path.join(dirpath, file + '_r' + str(num_of_hours) + '_m' + str(num_of_months) + '_w' + str(
            num_of_weeks)) + "_f" + str(num_for_predict)
        print('save file:', filename)
        np.savez_compressed(filename,
                            train=all_data['train'],
                            val=all_data['val'],
                            test=all_data['test'],
                            train_week=all_data["train"]["week"],
                            train_month=all_data['train']['month'],
                            train_recent=all_data['train']['recent'],
                            train_target=all_data['train']['target'],
                            val_week=all_data["val"]["week"],
                            val_month=all_data['val']['month'],
                            val_recent=all_data['val']['recent'],
                            val_target=all_data['val']['target'],
                            test_week=all_data["test"]["week"],
                            test_month=all_data['test']['month'],
                            test_recent=all_data['test']['recent'],
                            test_target=all_data['test']['target'],
                            stats=all_data['stats']
                            )

    return all_data

def normalization(train, val, test):
    """
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    """
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    mean = train.mean(axis=(0, 1, 3), keepdims=True)
    std = train.std(axis=(0, 1, 3), keepdims=True)
    print('mean.shape:', mean.shape)
    print('std.shape:', std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm


# all_data = read_and_generate_dataset("../data/Manchester/ManchesterDataFinall.npz", 1, 1, 1,  # 预测未来一小时分钟的
#                                      4,
#                                      points_per_hour=4,
#                                      times=96,
#                                      merge=False,
#                                      save=True)

all_data = read_and_generate_dataset("../data/PEMS08/PEMS08.npz", 1, 1, 1,  # 预测未来一小时分钟的
                                     12,
                                     points_per_hour=12,
                                     times=288,
                                     merge=False,
                                     save=True)
