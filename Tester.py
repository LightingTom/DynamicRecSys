import pickle


class Tester:

    def __init__(self, path):
        # Top K
        self.k = [5, 10, 20]

        self.session_length = 20 - 1
        self.pickle_path = path
        self.min_time = 1.0
        self.i_count = [0] * self.session_length
        self.recall = [[0] * len(self.k) for _ in range(self.session_length)]
        self.mrr = [[0] * len(self.k) for _ in range(self.session_length)]

        # 特殊对待第一项（因为第一项是根据inter_rnn给出的hidden state作为输入）
        self.first_recall = [0] * len(self.k)
        self.first_mrr = [0] * len(self.k)
        self.first_count = 0

        # 用于记录不同的范围的time gap结果的精确度
        self.time_buckets = [self.min_time, 2, 12, 36, 60, 84, 108, 132, 156, 180, 204, 228, 252, 276, 300, 348, 396,
                             444, 500, 501]
        self.time_count = [0] * len(self.time_buckets)
        self.mae = [0] * len(self.time_buckets)
        self.time_percent_mae = [0] * len(self.time_buckets)

    def reinitialize(self):
        self.i_count = [0] * self.session_length
        self.recall = [[0] * len(self.k) for _ in range(self.session_length)]
        self.mrr = [[0] * len(self.k) for _ in range(self.session_length)]
        self.first_recall = [0] * len(self.k)
        self.first_mrr = [0] * len(self.k)
        self.first_count = 0
        self.time_buckets = [self.min_time, 2, 12, 36, 60, 84, 108, 132, 156, 180, 204, 228, 252, 276, 300, 348, 396,
                             444, 500, 501]
        self.time_count = [0] * len(self.time_buckets)
        self.mae = [0] * len(self.time_buckets)
        self.time_percent_mae = [0] * len(self.time_buckets)

    @staticmethod
    def get_rank(target, prediction):
        for i in range(len(prediction)):
            if target == prediction[i]:
                return i + 1
        return -1

    def evaluate_sequence(self, predicted_sequence, target_sequence, seq_len):
        for i in range(seq_len):
            target = target_sequence[i]
            prediction = predicted_sequence[i]
            for j in range(len(self.k)):
                if target in prediction.data[:self.k[j]]:
                    self.recall[i][j] += 1
                    self.mrr[i][j] += 1.0 / self.get_rank(target, prediction.data[:self.k[j]])
            self.i_count[i] += 1

    def evaluate_first_item(self, predictions, target):
        for i in range(len(self.k)):
            if target in predictions.data[: self.k[i]]:
                self.first_recall[i] += 1
                self.first_mrr[i] += 1.0 / self.get_rank(target, predictions.data[: self.k[i]])
        self.first_count += 1

    def evaluate_batch(self, predictions, targets, seq_len, first_predictions, first_targets):
        for index in range(predictions):
            self.evaluate_sequence(predictions[index], targets[index], seq_len[index])
            self.evaluate_first_item(first_predictions[index], first_targets[index])

    def evaluate_time(self, time_prediction, time_target):
        for i in range(len(self.time_buckets)):
            if time_target < self.time_buckets[i] or i == len(self.time_buckets) - 1:
                self.time_count[i] += 1
                self.mae[i] += abs(time_prediction - time_target)
                # 忽略那些半小时以内的time gap(可近似的认为是同一session的操作)
                if time_target >= 0.5:
                    self.time_percent_mae[i] += 100 * (self.mae[i] / time_target)
                break

    def evaluate_batch_time(self, time_predictions, time_targets):
        for i in range(len(time_targets)):
            self.evaluate_time(time_predictions.data[i].item(), time_targets[i])

    def get_recall_mrr_result(self):
        res = "Cumulative\n"
        res += "Recall@5\tRecall@10\tRecall@20\tMRR@5\tMRR@10\tMRR@20\n"

        # 特殊对待每隔session的第一个交互
        res += "First Item\n"
        recall_info = ""
        mrr_info = ""
        for i in range(len(self.k)):
            recall_num = self.first_recall[i] / self.first_count
            mrr_num = self.first_mrr[i] / self.first_count
            recall_info += str(round(recall_num, 4)) + '\t'
            mrr_info += str(round(mrr_num, 4)) + '\t'
        res += recall_info + mrr_info + '\n'

        recall = [0] * len(self.k)
        mrr = [0] * len(self.k)
        count = 0
        for i in range(self.session_length):
            recall_info = ""
            mrr_info = ""
            res += "i <= " + str(i + 1) + "\n"
            # 考虑前面所有i个位置的效果，取其平均
            count += self.i_count[i]
            for j in range(len(self.k)):
                recall[j] += self.recall[i][j]
                mrr[j] += self.mrr[i][j]
                recall_info += str(round(recall[j] / count, 4)) + "\t"
                mrr_info += str(round(mrr[j] / count, 4)) + "\t"
            res += recall_info + mrr_info + "\n"
        return res

    def get_individual_recall_mrr_result(self):
        res = "Individual\n"
        res += "Recall@5\tRecall@10\tRecall@20\tMRR@5\tMRR@10\tMRR@20\n"
        for i in range(self.session_length):
            recall_info = ""
            mrr_info = ""
            res += "i <= " + str(i + 1) + "\n"
            for j in range(len(self.k)):
                recall_info += str(round(self.recall[i][j] / self.i_count[i], 4)) + "\t"
                mrr_info += str(round(self.mrr[i][j] / self.i_count[i], 4)) + "\t"
            res += recall_info + mrr_info + "\n"
        return res

    def get_time_result(self):
        res = "MAE\tPercent"
        total_count = 0
        total_mae = 0
        total_percent_mae = 0

        for i in range(len(self.time_buckets)):
            total_count += max(self.time_count[i], 1)
            total_mae += self.mae[i]
            total_percent_mae += self.time_percent_mae[i]

            res += "days<=" + self.time_buckets[i] + "\n"
            res += str(round(self.mae[i] / max(self.time_count[i], 1), 4)) + "\t"
            res += str(round(self.time_percent_mae[i] / max(self.time_count[i], 1), 4)) + "\n"
        res += "Total:\n"
        res += str(round(total_mae / max(total_count, 1), 4)) + "\t"
        res += str(round(total_percent_mae / max(total_count, 1), 4)) + "\n"
        return res

    def get_result(self, if_get_time=True):
        cumulate_recall_mrr_res = self.get_recall_mrr_result()
        individual_recall_mrr_res = self.get_individual_recall_mrr_result()
        time_res = ""
        if if_get_time:
            time_res = self.get_time_result()
        return cumulate_recall_mrr_res, time_res, individual_recall_mrr_res

    def store_result(self):
        mrr_recall_res = {"i_count": self.i_count, "k": self.k, "session_length": self.session_length, "mrr": self.mrr,
                          "recall": self.recall, "first_count": self.first_count, "first_mrr": self.first_mrr,
                          "first_recall": self.first_recall}
        time_res = {"mae": self.mae, "count": self.time_count, "time_buckets": self.time_buckets,
                    "time_percent_mae": self.time_percent_mae}
        res = {"mrr_recall_res": mrr_recall_res, "time_res": time_res}
        pickle.dump(res, open(self.pickle_path + ".pickle", "wb"))

    def get_result_and_reset(self, store=True):
        res = self.get_result()
        if store:
            self.store_result()
        self.reinitialize()
        return res
