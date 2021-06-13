# -*- coding: utf-8 -*-
r"""
    Author: Nicolas Zampieri
    Date: September 03, 2020
"""

import argparse
import csv
import numpy as np
from tqdm import tqdm

def load_universal_sentence_embedder(X):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow_hub as hub
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module(module_url)
    embeddings = embed(X)

    session = tf.Session()
    tf.keras.backend.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())

    return session.run(embeddings)


def write_embeddings(embeddings, path_file):
    writer = csv.writer(open(path_file, 'w', encoding="utf-8"), delimiter="\t")
    for vector in embeddings:
        writer.writerow(vector)


def print_embeddings(embeddings):
    for key, vector in embeddings:
        print(key, vector)


def load_embeddings(path_file, size=512):
    reader = csv.reader(open(path_file, 'r', encoding="utf-8"), delimiter="\t")
    embeddings = []
    for row in reader:
        vector = np.zeros(size)
        assert len(row) == size
        for v_index in range(len(row)):
            vector[v_index] = float(row[v_index])
        embeddings.append(vector)

    return np.array(embeddings)


def align_key_embeddings(key, embeddings):
    embed_aligned = []
    for index in range(len(key)):
        embed_aligned.append((key[index], embeddings[index]))
    return embed_aligned


def read_file(path_file, is_founta):
    file = open(path_file, 'r', encoding="UTF-8")

    tweets = []
    if is_founta:
        first_row = False  # To read Founta corpus
        index_tweet = 0
        reader = csv.reader(file, delimiter='\t')
    else:
        first_row = True  # To read Hateval2019 corpus
        index_tweet = 1
        reader = csv.reader(file, delimiter=',')
    for row in reader:
        if first_row:
            first_row = False
        else:
            tweet = row[index_tweet]
            tweets.append(tweet)
    file.close()
    return tweets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", dest="input_file", type=str, required=True,
                        help="Path for input file.")
    parser.add_argument("--output_file", dest="output_file", type=str, required=True,
                        help="Path to save use embeddings file (Extension with .usembed.")
    parser.add_argument("--per_batch", dest="per_batch", type=int, required=False, default=1000,
                        help="To use batch_size.")
    parser.add_argument("--founta", dest="founta", type=bool, required=False, nargs='?',
                        const=True,
                        help="""Option to use founta dataset.
                             Refer to --help to see how argument used.
                             """)
    args = parser.parse_args()

    tweets = read_file(args.input_file, args.founta)
    embeddings_full = []
    for batch_size in tqdm(range(int(len(tweets) / args.per_batch) + 1)):
        batch_begin = batch_size * args.per_batch
        batch_end = (batch_size + 1) * args.per_batch
        if batch_end > len(tweets):
            embeddings = load_universal_sentence_embedder(tweets[batch_begin:])
        elif batch_begin == 0:
            embeddings = load_universal_sentence_embedder(tweets[:batch_end])
        else:
            embeddings = load_universal_sentence_embedder(tweets[batch_begin:batch_end])
        for embed in embeddings:
            embeddings_full.append(embed)
    write_embeddings(embeddings=embeddings_full, path_file=args.output_file)

    assert len(load_embeddings(path_file=args.output_file)) == len(embeddings_full)
