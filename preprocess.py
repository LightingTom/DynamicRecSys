import dateutil.parser
import pickle
import os
import time
import numpy as np

runtime = time.time()
reddit = "subreddit"
dataset = reddit

home = os.path.expanduser('~')

# optional settings
MINIMUM_STD = 8
SPLIT_ON_TIMESTAMP = False

split_mode = "user"

DATASET_DIR = '/home/dl2021_3/Recsys/datasets'

DATASET_FILE = DATASET_DIR + '/reddit_data.csv'
DATASET_W_CONVERTED_TIMESTAMPS = DATASET_DIR + '/1_converted_timestamps.pickle'
DATASET_USER_ARTIST_MAPPED = DATASET_DIR + '/2_user_artist_mapped.pickle'
DATASET_USER_SESSIONS = DATASET_DIR + '/3_user_sessions.pickle'
DATASET_TRAIN_TEST_SPLIT = DATASET_DIR + '/4_train_test_split_' + split_mode + '.pickle'
DATASET_USER_REDUCE = DATASET_DIR + '/5_train_test_split_' + split_mode + '.pickle'
DATASET_BPR_MF = DATASET_DIR + '/bpr-mf_train_test_split.pickle'
SESSION_TIMEDELTA = 60 * 60  # 1 hour

MAX_SESSION_LENGTH = 20  # maximum number of events in a session
MAX_SESSION_LENGTH_PRE_SPLIT = MAX_SESSION_LENGTH * 2
MINIMUM_REQUIRED_SESSIONS = 3  # The dual-RNN should have minimum 2 two train + 1 to test
PAD_VALUE = 0
SPLIT_FRACTION = 0.8


def file_exists(filename):
    return os.path.isfile(filename)


def load_pickle(pickle_file):
    return pickle.load(open(pickle_file, 'rb'))


def save_pickle(data_object, data_file):
    pickle.dump(data_object, open(data_file, 'wb'))


'''
simply read the data from the csv file 
and then reverse the list to get the data nearest to us in the front
'''


def convert_timestamps_reddit():
    dataset_list = []
    with open(DATASET_FILE, 'rt', buffering=10000, encoding='utf8') as dataset:
        for line in dataset:
            line = line.rstrip()
            line = line.split(',')
            if line[2] == 'utc':
                continue
            user_id = line[0]
            subreddit = line[1]
            timestamp = float(line[2])
            dataset_list.append([user_id, timestamp, subreddit])

    dataset_list = list(reversed(dataset_list))

    save_pickle(dataset_list, DATASET_W_CONVERTED_TIMESTAMPS)


def convert_timestamps_lastfm():
    dataset_list = []
    with open(DATASET_FILE, 'rt', buffering=10000, encoding='utf8') as dataset:
        for line in dataset:
            line = line.split('\t')
            user_id = line[0]
            timestamp = (dateutil.parser.parse(line[1])).timestamp()
            artist_id = line[2]
            dataset_list.append([user_id, timestamp, artist_id])

    dataset_list = list(reversed(dataset_list))

    save_pickle(dataset_list, DATASET_W_CONVERTED_TIMESTAMPS)


'''
Replace the user_id and artist_id in last_fm with index
'''


def map_user_and_artist_id_to_labels():
    dataset_list = load_pickle(DATASET_W_CONVERTED_TIMESTAMPS)
    artist_map = {}
    user_map = {}
    artist_id = ''
    user_id = ''
    for i in range(len(dataset_list)):
        user_id = dataset_list[i][0]
        artist_id = dataset_list[i][2]

        # use the length of map now as the label
        if user_id not in user_map:
            user_map[user_id] = len(user_map)
        if artist_id not in artist_map:
            artist_map[artist_id] = len(artist_map)

        dataset_list[i][0] = user_map[user_id]
        dataset_list[i][2] = artist_map[artist_id]

    # Save to pickle file
    save_pickle(dataset_list, DATASET_USER_ARTIST_MAPPED)


def split_single_session(session):
    # first divide into sessions with MAX_SESSION_LENGTH
    splitted = [session[i:i + MAX_SESSION_LENGTH] for i in range(0, len(session), MAX_SESSION_LENGTH)]
    if len(splitted[-1]) < 2:
        del splitted[-1]
    # TODO: propose?
    first_time = splitted[0][0][0]
    last_time = splitted[-1][-1][0]
    for session in splitted:
        session[0][0] = first_time
        session[-1][0] = first_time
    splitted[-1][-1][0] = last_time

    return splitted


def perform_session_splits(sessions):
    splitted_sessions = []
    for session in sessions:
        splitted_sessions += split_single_session(session)

    return splitted_sessions


def split_long_sessions(user_sessions):
    for k, v in user_sessions.items():
        user_sessions[k] = perform_session_splits(v)


def collapse_session(session):
    new_session = [session[0]]
    for i in range(1, len(session)):
        last_event = new_session[-1]
        current_event = session[i]
        # TODO: Can we surely know the time is in order
        if current_event[1] != last_event[1]:
            new_session.append(current_event)
    return new_session


def collapse_repeating_items(user_sessions):
    for k, sessions in user_sessions.items():
        for i in range(len(sessions)):
            sessions[i] = collapse_session(sessions[i])


''' Splits sessions according to inactivity (time between two consecutive 
    actions) and assign sessions to their user. Sessions should be sorted, 
    both eventwise internally and compared to other sessions, but this should 
    be automatically handled since the dataset is presorted
'''


def sort_and_split_usersessions():
    dataset_list = load_pickle(DATASET_USER_ARTIST_MAPPED)
    user_sessions = {}
    current_session = []
    for event in dataset_list:
        user_id = event[0]
        timestamp = event[1]
        artist = event[2]

        new_event = [timestamp, artist]

        # if new user -> new session
        if user_id not in user_sessions:
            user_sessions[user_id] = []
            current_session = []
            user_sessions[user_id].append(current_session)
            current_session.append(new_event)
            continue

        # it is an existing user: is it a new session?
        # we also know that the current session contains at least one event
        # NB: Dataset is presorted from newest to oldest events
        last_event = current_session[-1]
        last_timestamp = last_event[0]
        timedelta = timestamp - last_timestamp

        if timedelta < SESSION_TIMEDELTA:
            # new event belongs to current session
            current_session.append(new_event)
        else:
            # new event belongs to new session
            # notice that the map store the address of the list, not a copy of the list
            current_session = [new_event]
            user_sessions[user_id].append(current_session)

    collapse_repeating_items(user_sessions)

    # Remove sessions that only contain one event
    # Bad to remove stuff from the lists we are iterating through, so create
    # a new data structure and copy over what we want to keep
    # The result will be sessions with length [2, MAX_SESSION_LENGTH_PRE_SPLIT)
    new_user_sessions = {}
    for k in user_sessions.keys():
        if k not in new_user_sessions:
            new_user_sessions[k] = []

        us = user_sessions[k]
        for session in us:
            if len(session) > 1 and len(session) < MAX_SESSION_LENGTH_PRE_SPLIT:
                new_user_sessions[k].append(session)

    # If it is longer than MAX_SESSION_LENGTH_PRE_SPLIT, just abandon it
    # Split too long sessions, before removing user with too few sessions
    # Because splitting can result in more sessions.

    split_long_sessions(new_user_sessions)

    # Remove users with less than 3 session
    # Find users with less than 3 sessions first
    to_be_removed = []
    for k, v in new_user_sessions.items():
        if len(v) < MINIMUM_REQUIRED_SESSIONS:
            to_be_removed.append(k)
    # Remove the users we found
    for user in to_be_removed:
        new_user_sessions.pop(user)

    # Do a remapping to account for removed data
    print("remapping to account for removed data...")

    # remap users
    nus = {}
    for k, v in new_user_sessions.items():
        nus[len(nus)] = new_user_sessions[k]

    # remap artistIDs
    art = {}
    for k, v in nus.items():
        for session in v:
            for i in range(len(session)):
                a = session[i][1]
                if a not in art:
                    art[a] = len(art) + 1
                session[i][1] = art[a]

    save_pickle(nus, DATASET_USER_SESSIONS)


# get each sessions' length
def get_session_lengths(dataset):
    session_lengths = {}
    for k, v in dataset.items():
        session_lengths[k] = []
        for session in v:
            session_lengths[k].append(len(session) - 1)

    return session_lengths


def create_padded_sequence(session):
    if len(session) == MAX_SESSION_LENGTH:
        return session
    # If a session's length is smaller than MAX_SESSION_LENGTH, then pad it with [0,0] (append at last)
    dummy_timestamp = 0
    dummy_label = 0
    length_to_pad = MAX_SESSION_LENGTH - len(session)
    padding = [[dummy_timestamp, dummy_label]] * length_to_pad
    session += padding
    return session


def pad_sequences(dataset):
    for k, v in dataset.items():
        for session_index in range(len(v)):
            dataset[k][session_index] = create_padded_sequence(dataset[k][session_index])


# Splits the dataset into a training and a testing set, by extracting the last
# sessions of each user into the test set
def split_to_training_and_testing():
    dataset = load_pickle(DATASET_USER_SESSIONS)
    trainset = {}
    testset = {}

    if (SPLIT_ON_TIMESTAMP):
        timestamps = []
        for k, v in dataset.items():
            for session in v:
                timestamps.append(session[0][0])
        n_sessions = len(timestamps)
        # not split according to the origin array, split using time instead
        split_time = sorted(timestamps)[int(SPLIT_FRACTION * n_sessions)]

        for k, v in dataset.items():
            split_point = len(v)
            for i in range(len(v)):
                if (v[i][0][0] > split_time):
                    split_point = i
            trainset[k] = v[:split_point]
            testset[k] = v[split_point:]

    else:
        for k, v in dataset.items():
            n_sessions = len(v)
            split_point = int(SPLIT_FRACTION * n_sessions)

            # runtime check to ensure that we have enough sessions for training and testing
            if split_point < 2:
                raise ValueError('User ' + str(k) + ' with ' + str(n_sessions) + """ sessions, 
                    resulted in split_point: '+str(split_point)+' which gives too 
                    few training sessions. Please check that data and preprocessing 
                    is correct.""")

            trainset[k] = v[:split_point]
            testset[k] = v[split_point:]

    # Also need to know session lengths for train- and testset
    train_session_lengths = get_session_lengths(trainset)
    test_session_lengths = get_session_lengths(testset)

    # Finally, pad all sequences before storing everything
    pad_sequences(trainset)
    pad_sequences(testset)

    # Put everything we want to store in a dict, and just store the dict with pickle
    pickle_dict = {}
    pickle_dict['trainset'] = trainset
    pickle_dict['testset'] = testset
    pickle_dict['train_session_lengths'] = train_session_lengths
    pickle_dict['test_session_lengths'] = test_session_lengths

    save_pickle(pickle_dict, DATASET_TRAIN_TEST_SPLIT)


def remove_noisy_users():
    dataset = load_pickle(DATASET_TRAIN_TEST_SPLIT)
    train = dataset["trainset"]
    print(len(train))
    test = dataset["testset"]
    train_lens = dataset["train_session_lengths"]
    test_lens = dataset["test_session_lengths"]

    # find the standard deviations of each users session gaps, session gaps of 0 are
    # the product of splitted session, and are therefore not considered
    stds = []
    for k in train.keys():
        user = []
        sessions = train[k]
        for i in range(1, len(sessions)):
            gap = (sessions[i][0][0] - sessions[i - 1][train_lens[k][i - 1]][0]) / 3600
            if (gap > 0):
                user.append(gap)
        if (len(train[k]) > 0 and len(test[k]) > 0):
            gap = (test[k][0][0][0] - sessions[-1][train_lens[k][-1]][0]) / 3600
            if (gap > 0):
                user.append(gap)
        sessions = test[k]
        for i in range(1, len(sessions)):
            gap = (sessions[i][0][0] - sessions[i - 1][test_lens[k][i - 1]][0]) / 3600
            if (gap > 0):
                user.append(gap)
        stds.append(np.std(user))

    # get the indices of users to be removed because of small stds
    stds = np.array(stds)
    remove = np.array(list(range(len(train))))[stds < MINIMUM_STD]

    # remove identified users from each dict
    for user in remove:
        train.pop(user)
        test.pop(user)
        train_lens.pop(user)
        test_lens.pop(user)

    print(len(train))
    # create new dicts for containing the users with updated IDs
    n_test = {}
    n_train = {}
    n_test_lens = {}
    n_train_lens = {}
    for k in train.keys():
        n_test[len(n_test)] = test[k]
        n_train[len(n_train)] = train[k]
        n_test_lens[len(n_test_lens)] = test_lens[k]
        n_train_lens[len(n_train_lens)] = train_lens[k]

    # remap artistIDs
    art = {}
    art[PAD_VALUE] = PAD_VALUE
    for k, v in n_train.items():
        for session in v:
            for i in range(len(session)):
                a = session[i][1]
                if a not in art:
                    art[a] = len(art)
                session[i][1] = art[a]
    # print(len(art))
    for k, v in n_test.items():
        for session in v:
            for i in range(len(session)):
                a = session[i][1]
                if a not in art:
                    art[a] = len(art)
                session[i][1] = art[a]
    # print(len(art))

    # create new dict for the resulting dataset
    pickle_dict = {}
    pickle_dict["trainset"] = n_train
    pickle_dict["testset"] = n_test
    pickle_dict["train_session_lengths"] = n_train_lens
    pickle_dict["test_session_lengths"] = n_test_lens

    save_pickle(pickle_dict, DATASET_USER_REDUCE)


def create_bpr_mf_sets():
    p = load_pickle(DATASET_TRAIN_TEST_SPLIT)
    train = p['trainset']
    train_sl = p['train_session_lengths']
    test = p['testset']
    test_sl = p['test_session_lengths']

    for user in train.keys():
        extension = test[user][:-1]
        # notice the difference between append and extend
        train[user].extend(extension)
        extension = test_sl[user][:-1]
        train_sl[user].extend(extension)

    for user in test.keys():
        test[user] = [test[user][-1]]
        test_sl[user] = [test_sl[user][-1]]

    pickle_dict = {}
    pickle_dict['trainset'] = train
    pickle_dict['testset'] = test
    pickle_dict['train_session_lengths'] = train_sl
    pickle_dict['test_session_lengths'] = test_sl

    save_pickle(pickle_dict, DATASET_BPR_MF)


if not file_exists(DATASET_W_CONVERTED_TIMESTAMPS):
    print("Converting timestamps.")
    convert_timestamps_reddit()

if not file_exists(DATASET_USER_ARTIST_MAPPED):
    print("Mapping user and artist IDs to labels.")
    map_user_and_artist_id_to_labels()

if not file_exists(DATASET_USER_SESSIONS):
    print("Sorting sessions to users.")
    sort_and_split_usersessions()

if not file_exists(DATASET_TRAIN_TEST_SPLIT):
    print("Splitting dataset into training and testing sets.")
    split_to_training_and_testing()

if not file_exists(DATASET_USER_REDUCE):
    print("Removing noisy users")
    remove_noisy_users()

"""
if not file_exists(DATASET_BPR_MF):
    print("Creating dataset for BPR-MF.")
    create_bpr_mf_sets()
"""

print("Runtime:", str(time.time() - runtime))
