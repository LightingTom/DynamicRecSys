import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from DataHandler import RNNDataHandler


# embedding layer
class Embedding(nn.Module):
    def __init__(self, input_dim, output_dim, item=False):
        super(Embedding, self).__init__()
        if item:
            self.embedding_table = nn.Embedding(input_dim, output_dim, padding_idx=0)
        else:
            self.embedding_table = nn.Embedding(input_dim, output_dim)

    def forward(self, x):
        return self.embedding_table(x)


class Intra_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(Intra_RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, lengths):
        x = self.dropout(x)
        gru_res, _ = self.gru(x, hidden)
        output = self.dropout(gru_res)
        output = self.linear(output)
        # the value in lengths is the index of the last element which is not padded 0
        idx = lengths.view(-1, 1, 1).expand(gru_res.size[0], 1, gru_res.size[2])
        # get rid of the padding zeros
        hidden_out = torch.gather(gru_res, 1, idx)
        hidden_out = hidden_out.squeeze().unsqueeze(0)
        return output, hidden_out


class Inter_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(Inter_RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x, hidden, idx):
        x = self.dropout(x)
        gru_res = self.gru(x, hidden)
        # Same as Intra_RNN
        hidden_idx = idx.view(-1, 1, 1).expand(gru_res.size(0), 1, gru_res.size(2))
        hidden_output = torch.gather(gru_res, 1, hidden_idx)
        hidden_output = hidden_output.squeeze().unsqueeze(0)
        hidden_output = self.dropout(hidden_output)
        return hidden_output

    def init_hidden(self, batch_size):
        return Variable(torch.zeros((1, batch_size, self.hidden_dim), dtype=torch.float))


class Time_Loss(nn.Module):
    def __init__(self):
        super(Time_Loss, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor([-0.1]))
        # self.w.data.uniform_(-0.1,0.1)

    def forward(self, time, target, epsilon):
        time_exp = torch.exp(time)
        w_target = self.w * torch.pow(target, epsilon)
        exps = (time_exp * (1 - torch.exp(w_target))) / self.w
        output = time + w_target + exps
        return -output

    def get_w(self):
        return self.w


class DynamicRecModel:
    def __init__(self, dim, dropout, params, flags, datahandler, tester, time_threshold):
        self.dim = dim
        self.params = params
        self.flags = flags
        self.datahandler = datahandler
        self.tester = tester
        self.time_threshold = time_threshold
        self.dropout = dropout
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.init_model()

    def init_model(self):
        model_params = []
        time_params = []
        # item embedding
        self.item_embed = Embedding(self.dim['N_ITEMS'], self.dim['EMBEDDING_DIM'], item=True)
        self.item_embed = self.item_embed.cuda()
        model_params += list(self.item_embed.parameters())
        # time embedding
        self.time_embed = Embedding(self.dim["TIME_RESOLUTION"], self.dim["TIME_HIDDEN"], item=False)
        self.time_embed = self.time_embed.cuda()
        model_params += list(self.time_embed.parameters())
        # user embedding
        self.user_embed = Embedding(self.dim['N_USERS'], self.dim['USERS_HIDDEN'], item=False)
        self.user_embed = self.user_embed.cuda()
        model_params += list(self.user_embed.parameters())
        # init inter and intra model
        self.inter_model = Inter_RNN(self.dim['INTER_INPUT'], self.dim['INTER_HIDDEN'], self.dropout['INTER'])
        self.intra_model = Intra_RNN(self.dim['EMBEDDING_DIM'], self.dim['INTRA_HIDDEN'], self.dim['N_ITEMS'],
                                     self.dropout['INTRA'])
        self.inter_model.cuda()
        self.intra_model.cuda()
        model_params += list(self.inter_model.parameters())
        model_params += list(self.intra_model.parameters())
        # linear layer for time
        self.time_linear = nn.Linear(self.dim['INTER_HIDDEN'], 1)
        self.time_linear = self.time_linear.cuda()
        time_params += [{"params": self.time_linear.parameters(), "lr": 0.1 * self.params["lr"]}]
        # linear layer for first inter_rnn layer
        self.first_linear = nn.Linear(self.dim["INTER_HIDDEN"], self.dim["N_ITEMS"])
        self.first_linear = self.first_linear.cuda()
        # time loss function
        self.time_loss_func = Time_Loss()
        self.time_loss_func = self.time_loss_func.cuda()
        time_params += [{"params": self.time_loss_func.parameters(), "lr": 0.1 * self.params["lr"]}]
        # Use Adam to do SGD
        self.model_optimizer = torch.optim.Adam(model_params, lr=self.params['lr'])
        self.time_optimizer = torch.optim.Adam(time_params, lr=self.params['lr'])
        self.first_rec_optimizer = torch.optim.Adam(self.first_linear.parameters(), lr=self.params['lr'])

    def get_time_loss_weight(self):
        return self.time_loss_func.get_w()

    # calculate t*f*(t) used for
    @staticmethod
    def calculate_func_val(t, time_exp, w):
        part1 = time_exp * torch.exp(t * w)
        part2 = torch.exp((time_exp - part1) / w)
        prob = part1 * part2
        return t * prob

    def time_pred(self, time, w):
        # integration settings
        # integration_count += 1
        precision = 3000
        T = 700  # time units
        part1 = 100
        part2 = 600
        # moving data structures to the  for efficiency
        T = torch.cuda.FloatTensor([T])
        dt1 = torch.cuda.FloatTensor([part1 / precision])
        dt2 = torch.cuda.FloatTensor([part2 / precision])
        part1 = torch.cuda.FloatTensor([part1])

        # integration loops
        time_exp = torch.exp(time)
        time_preds1 = self.calculate_func_val(part1, time_exp, w)
        time_preds2 = self.calculate_func_val(T, time_exp, w) + time_preds1
        for i in range(1, precision // 2):  # high resolution loop
            t = (2 * i - 1) * dt1
            time_preds1 += 4 * self.calculate_func_val(t, time_exp, w)
            time_preds1 += 2 * self.calculate_func_val(t + dt1, time_exp, w)
        time_preds1 += 4 * self.calculate_func_val(part1 - dt1, time_exp, w)
        for i in range(1, precision // 2):  # rough resolution loop
            t = (2 * i - 1) * dt2 + part1
            time_preds2 += 4 * self.calculate_func_val(t, time_exp, w)
            time_preds2 += 2 * self.calculate_func_val(t + dt2, time_exp, w)
        time_preds2 += 4 * self.calculate_func_val(T - dt2, time_exp, w)

        # division moved to the end for efficiency
        time_preds1 *= dt1 / 3
        time_preds2 *= dt2 / 3

        return time_preds1 + time_preds2

    # switch between train mode and eval mode
    def train_mode(self):
        self.inter_model.train()
        self.intra_model.train()

    def eval_mode(self):
        self.inter_model.eval()
        self.intra_model.eval()

    # move batch data to cuda tensors
    def process_batch_inputs(self, items, session_reps, sess_time_reps, user_list):
        sessions = Variable(torch.cuda.FloatTensor(session_reps))
        items = Variable(torch.cuda.LongTensor(items))
        sess_gaps = Variable(torch.cuda.LongTensor(sess_time_reps))
        users = Variable(torch.cuda.LongTensor(user_list.tolist()))
        return items, sessions, sess_gaps, users

    def process_batch_targets(self, item_targets, time_targets, first_rec_targets):
        item_targets = Variable(torch.cuda.LongTensor(item_targets))
        time_targets = Variable(torch.cuda.FloatTensor(time_targets))
        first = Variable(torch.cuda.LongTensor(first_rec_targets))
        return item_targets, time_targets, first

    def train_on_batch(self, items, session_reps, sess_time_reps, user_list, item_targets, time_targets,
                       first_rec_targets, session_lengths, session_rep_lengths):
        """
        :param items: The session information, which contains artist_id in each session
        :param session_reps: The context representation of session
        :param sess_time_reps: The context representation of session gap time
        :param user_list: The user_id with remain sessions
        :param item_targets: The target for item output, artist_id too
        :param time_targets: The target for time output
        :param first_rec_targets: The first artist_id of each session, target for first inter_RNN output
        :param session_lengths: The length of each session
        :param session_rep_lengths: The length of each session_rep
        :return: Loss of current batch
        """
        # clear gradients
        self.model_optimizer.zero_grad()
        self.time_optimizer.zero_grad()
        self.first_rec_optimizer.zero_grad()

        # Get train data into cuda
        X, sessions, session_gaps, users = self.process_batch_inputs(items, session_reps, sess_time_reps, user_list)
        Y, T_targets, first_targets = self.process_batch_targets(item_targets, time_targets, first_rec_targets)

        # embed users and session_gaps
        embedded_session_gaps = self.time_embed(session_gaps)
        embedded_users = self.user_embed(users)
        embedded_users = embedded_users.unsqueeze(1)
        embedded_users = embedded_users.expand(embedded_users.size(0),
                                               embedded_session_gaps.size(1), embedded_users.size(2))

        # Do a inter_RNN first, get the initial input for intra_RNN
        rep_idx = Variable(torch.cuda.LongTensor(session_rep_lengths)) - 1
        inter_hidden = self.inter_model.init_hidden(sessions.size(0))
        inter_last_hidden = self.inter_model(torch.cat((sessions, embedded_session_gaps, embedded_users), 2)
                                             , inter_hidden, rep_idx)

        # get time score and first prediction scores from the first inter_RNN
        times = self.time_linear(inter_last_hidden).squeeze()
        first_pred = self.first_linear(inter_last_hidden).squeeze()

        # embed items
        embedded_X = self.item_embed(X)

        lengths = Variable(torch.cuda.FloatTensor(session_lengths).view(-1,1))
        sum_X = embedded_X.sum(1)
        mean_X = sum_X.div(lengths)

        # -1 to get the idx of the array
        lengths = lengths.long() - 1

        # Intra_RNN
        recommend_output, intra_hidden_out = self.intra_model(embedded_X, inter_hidden, lengths)

        # store the new session representation
        self.datahandler.store_user_session_representations(intra_hidden_out.data[0], user_list, time_targets)

        # calculate loss
        reshaped_Y = Y.view(-1)
        reshaped_rec_output = recommend_output.view(-1, self.dim["N_ITEMS"])
        rec_loss = self.cross_entropy_loss(reshaped_rec_output, reshaped_Y)
        sum_loss = rec_loss.sum(0)
        divident = Variable(torch.cuda.FloatTensor([sum(session_lengths)]))
        mean_loss = sum_loss / divident
        first_loss = self.cross_entropy_loss(first_pred, first_targets)
        sum_first_loss = first_loss.sum(0)
        mean_first_loss = sum_first_loss / embedded_X.size(0)
        time_loss = self.time_loss_func(times, T_targets, self.params['EPSILON'])
        mask = Variable(T_targets.data.ge(self.time_threshold).float())
        time_loss = time_loss * mask
        non_zero_count = 0
        for sign in mask.data:
            if sign != 0:
                non_zero_count += 1
        time_loss_divisor = Variable(torch.cuda.FloatTensor([max(non_zero_count, 1)]))
        mean_time_loss = time_loss.sum(0) / time_loss_divisor
        combined_loss = self.params["ALPHA"] * mean_time_loss + self.params["BETA"] * mean_loss + self.params[
            "GAMMA"] * mean_first_loss
        combined_loss.backward()

        # update parameters
        self.time_optimizer.step()
        self.first_rec_optimizer.step()
        self.model_optimizer.step()
        return mean_loss.data[0]

    def predict_on_batch(self, items, session_reps, sess_time_reps, user_list, item_targets, time_targets,
                         first_rec_targets, session_lengths, session_rep_lengths, time_error):
        # Get batch data into cuda
        X, sessions, session_gaps, users = self.process_batch_inputs(items, session_reps, sess_time_reps, user_list)

        # embed time gaps and users
        embedded_session_gaps = self.time_embed(session_gaps)
        embedded_users = self.user_embed(users)
        embedded_users = embedded_users.unsqueeze(1)
        embedded_users = embedded_users.expand(embedded_users.size(0),
                                               embedded_session_gaps.size(1), embedded_users.size(2))

        # get idx of non_padding elements
        rep_idx = Variable(torch.cuda.LongTensor(session_rep_lengths)) - 1

        # inter_RNN
        inter_hidden = self.inter_model.init_hidden(sessions.size(0))
        inter_last_hidden = self.inter_model(torch.cat((sessions, embedded_session_gaps, embedded_users), 2), inter_hidden, rep_idx)

        # get time score and first prediction scores from the inter_RNN
        times = self.time_linear(inter_last_hidden).squeeze()
        first_pred = self.first_linear(inter_last_hidden).squeeze()
        if time_error:
            w = self.time_loss_func.get_w()
            time_pred = self.time_pred(times.data, w.data)
            self.tester.evaluate_batch_time(time_pred, time_targets)

        # embed items
        embedded_X = self.item_embed(X)

        lengths = Variable(torch.cuda.FloatTensor(session_lengths).view(-1,1))
        sum_X = embedded_X.sum(1)
        mean_X = sum_X.div(lengths)

        # -1 get the array index
        lengths = lengths.long() - 1

        # call forward on the intra RNN
        recommend_output, intra_hidden_out = self.intra_rnn(embedded_X, inter_last_hidden, lengths)
        self.datahandler.store_user_session_representations(intra_hidden_out.data[0], user_list, time_targets)
        k_values, k_predictions = torch.topk(torch.cat((first_pred.unsqueeze(1), recommend_output), 1),
                                             self.params["TOP_K"])
        return k_predictions
