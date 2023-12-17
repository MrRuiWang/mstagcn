import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Transformer import Transformer
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(nn.Module):
    def __init__(self, args, dim_in, dim_out, cheb_k=2):
        super(GCN, self).__init__()
        self.cheb_k = cheb_k
        self.node_num = args.num_nodes
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.embed_dim = args.embed_dim
        self.weights_pool = nn.Parameter(torch.FloatTensor(args.embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(args.embed_dim, dim_out))
        self.update = nn.Parameter(torch.randn(args.num_nodes, args.num_nodes), requires_grad=True)

        self.fc = nn.Sequential(
            OrderedDict([('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                         ('sigmoid1', nn.Sigmoid()),
                         ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                         ('sigmoid2', nn.Sigmoid()),
                         ('fc3', nn.Linear(self.middle_dim, self.embed_dim))]))

    @staticmethod
    def get_laplacian(graph, I, normalize=True):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            # L = I - torch.matmul(torch.matmul(D, graph), D)
            L = torch.matmul(torch.matmul(D, graph), D)
        else:
            graph = graph + I
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        return L

    def forward(self, input, node_embeddings):
        # x B N F T
        # node_embeddings (B T N D) ( N D)
        filter = self.fc(input.permute(0, 1, 3, 2))  # (B N F T) -> (B N T D)
        filter = filter.permute(0, 2, 1, 3)  # (B N T D) -> (B T N D)

        batch_size, node_num, in_channels, num_of_timesteps = input.shape
        supports1 = torch.eye(node_num).to(node_embeddings[0].device)  # (N N )
        nodevec = torch.tanh(torch.mul(node_embeddings[0], filter))  # (B T N D)(B T N D)-> (B T N D)  [B,N,dim_in]
        supports2 = F.relu(torch.matmul(nodevec, nodevec.transpose(3, 2)))

        A_s = F.relu(torch.tanh(torch.matmul(node_embeddings[1], node_embeddings[1].transpose(1, 0))))  # 不需要属于（0-1）
        update = torch.sigmoid(self.update * (supports2 + A_s))
        supports2 = update * supports2 + (1 - update) * A_s

        supports2 = self.get_laplacian(supports2, supports1)  # (B T N D)(B T D N) -> (B T N N) （这一步会操作的）

        # (N N)
        x_g1 = torch.einsum("nm,btmc->btnc", supports1, input.permute(0, 3, 1, 2))  # B T N F
        x_g2 = torch.einsum("btnm,btmc->btnc", supports2, input.permute(0, 3, 1, 2))  # B T N F
        x_g = torch.stack([x_g1, x_g2], dim=1)  # (B 2 T N F)

        weights = torch.einsum('nd,dkio->nkio', node_embeddings[1],
                               self.weights_pool)  # (N D) (D 2 F dim_out) -> (N 2 F dim_out)

        bias = torch.matmul(node_embeddings[1], self.bias_pool)  # (N D)(D dim_out)-> (N,dim_out)

        x_g = x_g.permute(0, 2, 3, 1, 4)  # B, T, N, cheb_k, dim_in
        x_gconv = torch.einsum('btnki,nkio->btno', x_g, weights) + bias  # b,t, N, dim_out

        return x_gconv.permute(0, 2, 3, 1)    # b,t, N, dim_out (B N dim_out t)


class getNodeEmb(nn.Module):
    def __init__(self, args):
        super(getNodeEmb, self).__init__()
        self.node_embeddings1 = nn.Parameter(torch.randn(args.num_nodes, args.embed_dim), requires_grad=True)
        self.T_i_D_emb = nn.Parameter(torch.empty(args.times, args.embed_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.embed_dim))
        self.args = args

    def forward(self, x):
        x_temp = x.permute(0, 3, 1, 2)
        node_embedding1 = self.node_embeddings1
        t_i_d_data = x_temp[..., 1]

        T_i_D_emb = self.T_i_D_emb[(t_i_d_data * self.args.times).type(torch.LongTensor)]
        node_embedding1 = torch.mul(node_embedding1, T_i_D_emb)

        d_i_w_data = x_temp[..., 2]
        D_i_W_emb = self.D_i_W_emb[(d_i_w_data).type(torch.LongTensor)]
        node_embedding1 = torch.mul(node_embedding1, D_i_W_emb)

        node_embeddings = [node_embedding1, self.node_embeddings1]

        return node_embeddings


class TCN(nn.Module):
    def __init__(self, args):
        super(TCN, self).__init__()
        self.node_embedding_t_1 = nn.Parameter(torch.FloatTensor(args.len_input, args.embed_dim), requires_grad=True)

    def forward(self, x):
        E_t_x = F.softmax(torch.relu(torch.matmul(self.node_embedding_t_1, self.node_embedding_t_1.transpose(1, 0))),
                          dim=1)
        x_TAt = torch.matmul(E_t_x, x.permute(0, 1, 3, 2)).permute(0, 1, 3,
                                                                   2)  # T T (B N T F) -> B N T F -> B N F T
        return x_TAt


class STGCN_block(nn.Module):

    def __init__(self, in_channels, nb_chev_filter, nb_time_filter, args):
        super(STGCN_block, self).__init__()
        self.args = args
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, 1),
                                   padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, 1))
        self.ln = nn.LayerNorm(nb_time_filter)

        self.tcn = TCN(args)
        self.GCN = GCN(args, in_channels, nb_chev_filter, args.cheb_k)

    def forward(self, x, node_embedding_s):
        """
        :param x: (batch_size, N, F_in, T) B N F T
        :return: (batch_size, N, nb_time_filter, T)
        node_embedding_s:  [node_embedding1, self.node_embeddings1]
        """
        x_TAt = self.tcn(x)

        gcn = self.GCN(x_TAt, node_embedding_s)

        # convolution along the time axis
        time_conv_output = self.time_conv(gcn.permute(0, 2, 1, 3))  # b n f t -> b f n t

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        return x_residual


class STGCN_module(nn.Module):

    def __init__(self, args, time_strides):
        """
        :param args:
        :param time_strides:
        :return output
        """
        super(STGCN_module, self).__init__()
        self.args = args
        self.BlockList = nn.ModuleList(
            [STGCN_block(args.in_channels, args.nb_chev_filter, args.nb_time_filter, args)])

        self.BlockList.extend(
            [STGCN_block(args.nb_time_filter, args.nb_chev_filter, args.nb_time_filter, args)
             for _ in
             range(args.nb_block - 1)])

        self.final_conv = nn.Conv2d(int(args.len_input / time_strides), args.horizon,
                                    kernel_size=(1, args.nb_time_filter))

        self.Transformer_week = Transformer(args.lag * args.num_of_weeks, args.d_model, args.dropout,
                                            args.lag * args.num_of_weeks, args.d_k, args.n_heads,
                                            args.d_v,
                                            args.d_ff, args.n_layers)
        self.Transformer_month = Transformer(args.lag * args.num_of_months, args.d_model, args.dropout,
                                             args.lag * args.num_of_months, args.d_k, args.n_heads,
                                             args.d_v,
                                             args.d_ff, args.n_layers)

        self.fc_week = nn.Linear(args.input_dim, args.d_model, bias=False)
        self.fc_month = nn.Linear(args.input_dim, args.d_model, bias=False)

        self.out_week = nn.Linear(args.d_model, args.output_dim, bias=False)
        self.out_month = nn.Linear(args.d_model, args.output_dim, bias=False)

        self.lag_week = nn.Conv2d(args.lag * args.num_of_weeks, args.horizon, kernel_size=(1, 1), bias=True)
        self.lag_month = nn.Conv2d(args.lag * args.num_of_months, args.horizon, kernel_size=(1, 1), bias=True)

        self.W_A = nn.Parameter(torch.FloatTensor(args.horizon, args.num_nodes, args.output_dim))
        self.W_B = nn.Parameter(torch.FloatTensor(args.horizon, args.num_nodes, args.output_dim))
        self.W_C = nn.Parameter(torch.FloatTensor(args.horizon, args.num_nodes, args.output_dim))

        self.node_emb = getNodeEmb(args)

    def forward(self, week_data, month_data, x):
        """
        :param x: (B, N, F, T)
        :param week_data: (B, T, N, F)
        :param month_data: (B, T, N, F)
        :return: (B, T, N, F)
        """

        week_data = week_data[:, :, :, 0:1]  # B T N F
        month_data = month_data[:, :, :, 0:1]  # B T N F
        batch_size = week_data.size(0)
        node_nums = week_data.size(2)
        seq_len_week = week_data.size(1)
        seq_len_month = month_data.size(1)
        week_data_temp = self.fc_week(week_data).permute(0, 2, 1, 3).reshape(batch_size * node_nums, seq_len_week,
                                                                             self.args.d_model)
        # (B, T_1, N, D)->(B, T_1, N, d_model)->(B, N, T_1, d_model)->(B * N, T_1, d_model)
        week_out = self.Transformer_week(week_data_temp)  # (B * N, T_1, d_model)-> （B * N, T_1, d_model）
        week_out = week_out.reshape(batch_size, node_nums, seq_len_week, -1).permute(0, 2, 1, 3)
        # （B * N, T_1, d_model）(B , N, T_1, d_model) (B , T_1, N,d_model)
        week_out = self.out_week(week_out)
        week_out = self.lag_week(week_out)  # (B , T_2, N, d_out)

        month_data_temp = self.fc_month(month_data).permute(0, 2, 1, 3).reshape(batch_size * node_nums, seq_len_month,
                                                                                self.args.d_model)
        # (B, T_1, N, D)->(B, T_1, N, d_model)->(B, N, T_1, d_model)->(B * N, T_1, d_model)
        month_out = self.Transformer_month(month_data_temp)  # (B * N, T_1, d_model)-> （B * N, T_1, d_model）
        #
        month_out = month_out.reshape(batch_size, node_nums, seq_len_month, -1).permute(0, 2, 1, 3)
        # （B * N, T_1, d_model）(B , N, T_1, d_model) (B , T_1, N,d_model)
        month_out = self.out_month(month_out)
        month_out = self.lag_month(month_out)

        node_embeddings = self.node_emb(x)

        # B N F T
        x = x[:, :, 0:1, :]
        for i in range(0, self.args.nb_block):
            x = self.BlockList[i](x, node_embeddings)
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        output = output.unsqueeze(-1)
        output = output.permute(0, 2, 1, 3)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        return output * self.W_A + week_out * self.W_B + month_out * self.W_C