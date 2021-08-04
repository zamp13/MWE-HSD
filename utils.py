import csv
import numpy as np
from tqdm import tqdm


def read_founta(path_file):
    reader = csv.reader(open(path_file, 'r', encoding='UTF-8'), delimiter='\t')
    TWEET = 0
    TARGET = 1
    tweets = []
    targets = []
    for row in reader:
        if row[TARGET] != 'spam':
            tweets.append(row[TWEET])
            targets.append(row[TARGET])
    vocab = {
        "<pad>": 0,
        "<unk>": 1
    }
    for tweet in tweets:
        for word in tweet.split():
            vocab[word] = len(vocab)

    vocab_target = {'normal': 0, 'abusive': 1, 'hateful': 2}
    targets_tokenize = []
    for target in targets:
        targets_tokenize.append(vocab_target[target])

    return tweets, targets_tokenize, vocab


def read_davidson(path_file):
    reader = csv.reader(open(path_file, 'r', encoding='UTF-8'), delimiter='\t')
    TWEET = 0
    TARGET = 1
    tweets = []
    targets = []
    for row in reader:
        if row[TARGET] != 'spam':
            tweets.append(row[TWEET])
            targets.append(int(row[TARGET]))
    vocab = {
        "<pad>": 0,
        "<unk>": 1
    }
    for tweet in tweets:
        for word in tweet.split():
            vocab[word] = len(vocab)

    return tweets, targets, vocab



def read_hateval(path_file):
    r"""

    :param with_preprocessing_crazytokenizer:
    :param path_file: String
    :return:
    """

    file = open(path_file, 'r', encoding="UTF-8")
    vocab = {
        "<pad>": 0,
        "<unk>": 1
    }
    X = []
    Y = []
    reader = csv.reader(file, delimiter=',')
    first_row = True
    for row in tqdm(reader):
        if first_row:
            first_row = False
        else:
            tweet = row[1]
            is_hateful = row[2]
            X.append(tweet)
            Y.append(int(is_hateful))
    file.close()
    for tweet in X:
        for word in tweet.split():
            vocab[word] = len(vocab)
    return X, Y, vocab


def plot_history_acc(history, path):
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.savefig(path)
    plt.close()


def plot_history_loss(history, path):
    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.savefig(path)
    plt.close()


def one_hot_weight(vocab):
    one_hot = []
    for k, v in vocab.items():
        if '<pad>' == 0:
            one_hot.append(np.zeros(len(vocab)))
        else:
            vector = np.zeros(len(vocab))
            vector[v - 1] = 1
            one_hot.append(vector)
    return np.array(one_hot)


def write_prediction_founta(path_file, prediction_file, prediction):
    r"""

    :param path_file:
    :param prediction:
    :return:
    """
    # path_file_write = path_file.split('.')[0] + '-ref.tsv'
    writer = csv.writer(open(prediction_file, 'w', encoding='utf-8'), delimiter='\t')
    # writer_ref = csv.writer(open(reference_file, 'w', encoding='utf-8'), delimiter='\t')
    reader = csv.reader(open(path_file, 'r', encoding='utf-8'), delimiter='\t')
    count_prediction = 0
    for row in reader:
        writer.writerow([row[0], prediction[count_prediction]])
        # writer_ref.writerow([row[0], row[2], row[3], row[4]])
        count_prediction += 1
    assert (count_prediction == len(prediction))


def write_prediction_hateval(path_file, prediction_file, prediction):
    r"""

    :param path_file:
    :param prediction:
    :return:
    """
    # path_file_write = path_file.split('.')[0] + '-ref.tsv'
    writer = csv.writer(open(prediction_file, 'w', encoding='utf-8'), delimiter='\t')
    # writer_ref = csv.writer(open(reference_file, 'w', encoding='utf-8'), delimiter='\t')
    reader = csv.reader(open(path_file, 'r', encoding='utf-8'), delimiter=',')
    count_prediction = 0
    first_row = True
    for row in reader:
        if first_row:
            first_row = False
            # writer.writerow([row[0], '{0,1}'])
        else:
            writer.writerow([row[0], str(prediction[count_prediction])])
            # writer_ref.writerow([row[0], row[2], row[3], row[4]])
            count_prediction += 1
    assert (count_prediction == len(prediction))


def prediction_to_class_softmax_founta(prediction):
    r"""
    :param prediction: list of probabilities.
    :return class_prediction: list of classes.
    """
    import numpy as np
    class_prediction = []
    for prob in prediction:
        prob_max = np.argmax(prob)
        if prob_max == 0:
            class_prediction.append('normal')
        if prob_max == 1:
            class_prediction.append('abusive')
        if prob_max == 2:
            class_prediction.append('hateful')
    return class_prediction


def prediction_to_class_softmax_davidson(prediction):
    r"""
    :param prediction: list of probabilities.
    :return class_prediction: list of classes.
    """
    import numpy as np
    class_prediction = []
    for prob in prediction:
        prob_max = np.argmax(prob)
        class_prediction.append(prob_max)
    return class_prediction


def prediction_to_class(prediction):
    r"""
    :param prediction: list of probabilities.
    :return class_prediction: list of classes.
    """
    class_prediction = []
    for prob in prediction:
        if prob < 0.5:
            class_prediction.append(0)
        else:
            class_prediction.append(1)
    return class_prediction


def reconstruct_sentence(tweet):
    tweets_reconstruct = []
    tweet_reconstruct = ""
    for word in tweet:
        tweet_reconstruct += word + " "
    return tweet_reconstruct
