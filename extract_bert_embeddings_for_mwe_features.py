# -*- coding: utf-8 -*-
r"""
    Author: Nicolas Zampieri
    Date: February 16, 2021
"""

# IMPORT
import argparse
from tqdm import tqdm
import mwe_features
import numpy as np


def align_for_lstm_models(vector_bert_mwe, size_features, size_embeddings):
    vector_final = []
    for index_value in range(size_features - len(vector_bert_mwe)):
        vector_final.append(np.zeros(size_embeddings))
    for embed_mwe in vector_bert_mwe:
        vector_final.append(embed_mwe)
    return vector_final


def align_mwe_with_bert_tokenisation(tweet, mwe_vector, embeddings_bert, tokenizer, size_embeddings=768,
                                     size_features=15, verbose=False):
    ### Input
    # orig_tokens = ["John", "Johanson", "'s", "house"]
    # labels = ["NNP", "NNP", "POS", "NN"]
    if isinstance(tweet, list):
        orig_tokens = tweet
    else:
        orig_tokens = tweet.split()

    ### Output
    bert_tokens = []

    # Token map will be an int -> int mapping between the `orig_tokens` index and
    # the `bert_tokens` index.
    orig_to_tok_map = []
    bert_tokens.append("[CLS]")
    for orig_token in orig_tokens:
        orig_to_tok_map.append(len(bert_tokens))
        bert_tokens.extend(tokenizer.tokenize(orig_token))
    bert_tokens.append("[SEP]")

    # bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
    # orig_to_tok_map == [1, 2, 4, 6]
    vector_bert_mwe = []
    if verbose:
        print(bert_tokens)
        print(orig_to_tok_map)
        print(mwe_vector[:len(orig_to_tok_map)])
    for index_mwe in range(len(orig_to_tok_map)):
        if mwe_vector[index_mwe] > 1:
            if index_mwe == len(orig_to_tok_map) - 1:
                if verbose:
                    print(bert_tokens[orig_to_tok_map[index_mwe]], mwe_vector[index_mwe])
                    print(embeddings_bert[orig_to_tok_map[index_mwe]]['token'])
                vector_bert_mwe.append(embeddings_bert[orig_to_tok_map[index_mwe]]['layers'][0]['values'])
            else:
                len_tokens = orig_to_tok_map[index_mwe + 1] - orig_to_tok_map[index_mwe]
                for token in list(range(len_tokens)):
                    # print(embeddings_bert[orig_to_tok_map[index_mwe] + len_token]['token'])
                    if verbose:
                        print(token, bert_tokens[orig_to_tok_map[index_mwe] + token], embeddings_bert[orig_to_tok_map[index_mwe] + token]['token'])
                    vector_bert_mwe.append(
                        np.array(embeddings_bert[orig_to_tok_map[index_mwe] + token]['layers'][0]['values']))
    vector_final = align_for_lstm_models(vector_bert_mwe, size_features, size_embeddings)

    return np.array(vector_final)


def load_all_embeddings_bert(bert_file):
    import json
    embeddings_tweet = []
    with open(bert_file, 'r') as json_file:
        json_list = list(json_file)
        for json_str in json_list:
            result = json.loads(json_str)
            embeddings_tweet.append(result['features'])
    return embeddings_tweet


def write_mwe_features_bert_embeddings(path_file, bert_embeddings):
    import json
    datas_embeddings = {}
    for features in bert_embeddings:
        feat = {}
        for vector_mwe in features:
            feat[len(feat)] = vector_mwe.tolist()
        datas_embeddings[len(datas_embeddings)] = feat
    with open(path_file, 'w') as file:
        json.dump(datas_embeddings, file)


def load_mwe_features_bert_embeddings(path_file):
    import json
    datas_embeddings = []
    with open(path_file, 'r') as file:
        file_embeddings = json.load(file)
        for index_tweet, feat in file_embeddings.items():
            array_feat = []
            for f, v in feat.items():
                array_feat.append(np.array(v))
            datas_embeddings.append(np.array(array_feat))
    print(np.array(datas_embeddings).shape)
    return np.array(datas_embeddings)


def read_bert_preprocess(input):
    file = open(input, 'r', encoding='UTF-8')
    line = file.readline()
    tweets = []
    while line:
        tweets.append(line)
        line = file.readline()
    return tweets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", type=str, required=True,
                        help="Input file (preprocess for BERT model) to extract BERT embeddinds for MWE.")
    parser.add_argument("--output", dest="output", type=str, required=True,
                        help="Output file to save BERT embeddings")
    parser.add_argument("--lexicon", dest="lexicon", type=str, required=True,
                        help="To load MWE lexicon")
    parser.add_argument("--bert_embeddings", dest="bert_embeddings", type=str, required=False,
                        help="Json file for embeddings bert. It use to extract vector of MWEs.")
    parser.add_argument("--vocab_bert", dest="vocab_bert", type=str, required=False,
                        help="Vocab bert file to load tokenizer")
    parser.add_argument("--with_discontinuity", dest="with_discontinuity", type=bool, required=False, const=True, nargs='?',
                        help="To use the possible discontinuity of MWE")
    parser.add_argument("--verbose", dest="verbose", type=bool, required=False, const=True, nargs='?',
                        help="To see the processing of extraction embeddings")
    args = parser.parse_args()

    tweets = read_bert_preprocess(args.input)
    lexicon = mwe_features.read_lexicon(args.lexicon)
    vocab_mwe, vocab_mwe_strong = mwe_features.tokenize_label_mwe(lexicon)
    if args.with_discontinuity:
        mwe_vectors, useless = mwe_features.annotated_corpus(tweets, lexicon, vocab_mwe, vocab_mwe_strong, 280)
    else:
        mwe_vectors, useless = mwe_features.annotated_corpus_with_discontinuity(tweets, lexicon, vocab_mwe, vocab_mwe_strong, 280)
    vectors_bert_embeddings = []
    from transformers import FullTokenizer
    tokenizer = FullTokenizer(vocab_file=args.vocab_bert, do_lower_case=True)
    embeddings_tweet = load_all_embeddings_bert(args.bert_embeddings)
    for index_tweet in tqdm(range(len(tweets))):
        vector_mwe = mwe_vectors[index_tweet]
        tweet = tweets[index_tweet]
        vector_bert_embeddings = align_mwe_with_bert_tokenisation(tweet, vector_mwe, embeddings_tweet[index_tweet], tokenizer,
                                                                  verbose=args.verbose, size_features=15)
        vectors_bert_embeddings.append(vector_bert_embeddings)
    write_mwe_features_bert_embeddings(args.output, vectors_bert_embeddings)
