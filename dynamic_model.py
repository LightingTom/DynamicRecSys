import torch
import time
import os

from datahandler import DataHandler
from Tester import Tester
from modules import DynamicRecModel

DATASET_DIR = ''  # fixme
DATASET_FILE = ''  # fixme

DATASET_W_CONVERTED_TIMESTAMPS = DATASET_DIR + '/1_converted_timestamps.pickle'
DATASET_USER_ARTIST_MAPPED = DATASET_DIR + '/2_user_artist_mapped.pickle'
DATASET_USER_SESSIONS = DATASET_DIR + '/3_user_sessions.pickle'
DATASET_TRAIN_TEST_SPLIT = DATASET_DIR + '/4_train_test_split_user.pickle'
DATASET_USER_REDUCE = DATASET_DIR + '/5_train_test_split_user.pickle'
DATASET_BPR_MF = DATASET_DIR + '/bpr-mf_train_test_split.pickle'

SESSION_TIMEDELTA = 3600  # 1h

MAX_SESSION_LENGTH = 20  # maximum number of events in a session
MAX_SESSION_LENGTH_PRE_SPLIT = MAX_SESSION_LENGTH * 2
MINIMUM_REQUIRED_SESSIONS = 3  # The dual-RNN should have minimum 2 two train + 1 to test
PAD_VALUE = 0
SPLIT_FRACTION = 0.8

SEED = 2
GPU = 0
gap_strat = ""

torch.manual_seed(SEED)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

dims = {}
params = {}
BATCHSIZE = 100
SEQLEN = 20 - 1
params["TOP_K"] = 20
MAX_SESSION_REPRESENTATIONS = 15
dims["TIME_RESOLUTION"] = 500
dims["TIME_HIDDEN"] = 5
dims["USER_HIDDEN"] = 10

dataset_path = ''  # fixme
txt_log_name = 'Log.txt'
with open(txt_log_name, 'w+') as txt_file:
    txt_file.write("New experiment\n")

dims["EMBEDDING_DIM"] = 50
params["lr"] = 0.001

dropout = {}
dropout["INTER"] = 0
dropout["INTRA"] = 0

MAX_EPOCHS = 30
min_time = 1.0
time_threshold = torch.cuda.FloatTensor([min_time]) / 24
dims["INTRA_HIDDEN"] = dims["EMBEDDING_DIM"]
dims["INTER_INPUT_DIM"] = dims["INTRA_HIDDEN"] + dims["TIME_HIDDEN"] + dims["USER_HIDDEN"]
dims["INTER_HIDDEN"] = dims["INTRA_HIDDEN"]

datahandler = DataHandler(dataset_path, BATCHSIZE, MAX_SESSION_REPRESENTATIONS, dims["INTRA_HIDDEN"],
                             dims["TIME_RESOLUTION"], min_time, gap_strat)
dims["N_ITEMS"] = datahandler.get_num_items()
N_SESSIONS = datahandler.get_num_training_sessions()
dims["N_USERS"] = datahandler.get_num_users()

# TODO: Initialize tester
tester = None
model = DynamicRecModel(dims, dropout, params, datahandler, tester, time_threshold)

# setting up for training
epoch_nr = 0
start_time = time.time()
num_training_batches = datahandler.get_num_training_batches()
num_test_batches = datahandler.get_num_test_batches()
epoch_loss = 0

# start training
while epoch_nr < MAX_EPOCHS:
    with open(txt_log_name, 'a') as txt_file:
        txt_file.write("Starting epoch #" + str(epoch_nr) + "\n")
    start_time_epoch = time.time()

    # reset the datahandler and get first training batch
    datahandler.reset_user_batch_data_train()
    datahandler.reset_user_session_representations()
    items, item_targets, session_lengths, session_reps, session_rep_lengths, user_list, sess_time_reps, time_targets, first_rec_targets = datahandler.get_next_train_batch()
    batch_nr = 0

    model.train_mode()
    while len(items) > int(BATCHSIZE / 2):
        batch_start_time = time.time()
        batch_loss = model.train_on_batch(items, session_reps, sess_time_reps, user_list, item_targets, time_targets,
                                          first_rec_targets, session_lengths, session_rep_lengths)
        epoch_loss += batch_loss
        batch_runtime = time.time() - batch_start_time
        items, item_targets, session_lengths, session_reps, session_rep_lengths, user_list, sess_time_reps, time_targets, first_rec_targets = datahandler.get_next_train_batch()
        if batch_nr % 1500 == 0:
            with open(txt_log_name, 'a') as txt_file:
                txt_file.write("Batch: " + str(batch_nr) + "/" + str(num_training_batches) + " batch_loss: " + str(
                    batch_loss) + "\n")
        batch_nr += 1
        with open(txt_log_name, 'a') as txt_file:
            txt_file.write("Epoch loss: " + str(epoch_loss / batch_nr) + "\n")
