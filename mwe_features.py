# IMPORT
# from new_reader import read_streusle_json

from utils import reconstruct_sentence
import spacy_udpipe

spacy_udpipe.download('en')
nlp = spacy_udpipe.load('en')

# DELETE_MWE = ["V.LVC.cause", "INF.P", "CCONJ", "SYM", "PRON", "NUM", "INTJ", "SCONJ"]
# VERBAL = ["V.IAV", "V.VPC.full", "V.VID", "V.VPC.semi", "V.LVC.full"]
# OTHER = ["N", "DISC", "PP", "ADV", "ADJ", "DET", "AUX", "P"]


def read_lexicon(path_file):
    import csv
    lexicon = []
    file = csv.reader(open(path_file, 'r', encoding='utf-8'), delimiter="\t")
    first_row = True
    len_lexicon = 0
    for row in file:
        if first_row:
            len_lexicon = int(row[0])
            first_row = False
        else:
            lexicon.append((row[0], row[1], row[2]))
    assert (len(lexicon) == len_lexicon)
    return lexicon


# def create_lexicon_from_json(datas, lexcat):
#     MWE = 'smwes'
#     VMWE = 'wmwes'
#     lexicon = []
#     lexicon_mwe = []
#
#     list_lexcat = []
#     if lexcat == 'all':
#         for cat in DELETE_MWE:
#             list_lexcat.append(cat)
#         for cat in VMWE:
#             list_lexcat.append(cat)
#         for cat in OTHER:
#             list_lexcat.append(cat)
#     elif lexcat == 'vmwe':
#         for cat in VMWE:
#             list_lexcat.append(cat)
#     elif lexcat == 'other':
#         for cat in OTHER:
#             list_lexcat.append(cat)
#     elif lexcat == 'mwe13':
#         for cat in VMWE:
#             list_lexcat.append(cat)
#         for cat in OTHER:
#             list_lexcat.append(cat)
#     else:
#         print("Error argument lexcat: " + lexcat)
#         exit(1001)
#
#     for corpus in datas:
#         for sentence in corpus:
#             for mwe, value in sentence[MWE].items():
#                 if value['lexlemma'] not in lexicon_mwe and value["lexcat"] in list_lexcat:
#                     lexicon_mwe.append(value['lexlemma'])
#                     if value["lexcat"] is not None and value["lexcat"] in list_lexcat:
#                         if value["ss"] is not None:
#                             lexicon.append((value['lexlemma'], value["lexcat"], value["ss"]))
#                         else:
#                             lexicon.append((value['lexlemma'], value["lexcat"], "null"))
#     return lexicon


# def search_mwe_in_tweet(tweet, hateful, lexicon, count_mwe, count_mwe_label):
#     tweet = tweet.lower()
#     count = 0
#     HATE = "hate"
#     NON_HATE = "non-hate"
#     GLOBAL = "global"
#     if GLOBAL not in count_mwe_label:
#         count_mwe_label[GLOBAL] = {}
#         count_mwe_label[HATE] = {}
#         count_mwe_label[NON_HATE] = {}
#     if GLOBAL not in count_mwe:
#         count_mwe[GLOBAL] = {}
#         count_mwe[HATE] = {}
#         count_mwe[NON_HATE] = {}
#     for mwe in lexicon:
#         if mwe[0] in tweet:
#             count += 1
#             # LABEL MWEs
#             if mwe[1] not in count_mwe_label[GLOBAL]:
#                 count_mwe_label[GLOBAL][mwe[1]] = 1
#                 count_mwe_label[HATE][mwe[1]] = 0
#                 count_mwe_label[NON_HATE][mwe[1]] = 0
#             else:
#                 count_mwe_label[GLOBAL][mwe[1]] += 1
#             # HATE TWEET
#             if hateful == 1:
#                 if mwe[1] not in count_mwe_label[HATE]:
#                     count_mwe_label[HATE][mwe[1]] = 1
#                 else:
#                     count_mwe_label[HATE][mwe[1]] += 1
#             else:
#                 # Non HATE TWEET
#                 if mwe[1] not in count_mwe_label[NON_HATE]:
#                     count_mwe_label[NON_HATE][mwe[1]] = 1
#                 else:
#                     count_mwe_label[NON_HATE][mwe[1]] += 1
#
#             if mwe[0] not in count_mwe[GLOBAL]:
#                 count_mwe[GLOBAL][mwe[0]] = 1
#             else:
#                 count_mwe[GLOBAL][mwe[0]] += 1
#             if hateful == 1:
#                 if mwe[0] not in count_mwe[HATE]:
#                     count_mwe[HATE][mwe[0]] = 1
#                 else:
#                     count_mwe[HATE][mwe[0]] += 1
#             else:
#                 if mwe[0] not in count_mwe[NON_HATE]:
#                     count_mwe[NON_HATE][mwe[0]] = 1
#                 else:
#                     count_mwe[NON_HATE][mwe[0]] += 1
#     return count


# def count_uniq_hate_nonhate_and_global_categories(lexicon, count_mwe, count_label):
#     HATE = "hate"
#     NON_HATE = "non-hate"
#     GLOBAL = "global"
#     count = {GLOBAL: {},
#              HATE: {},
#              NON_HATE: {}}
#     for label in count_label[GLOBAL]:
#         count[GLOBAL][label] = 0
#         count[HATE][label] = 0
#         count[NON_HATE][label] = 0
#     for mwe in lexicon:
#         if mwe[0] in count_mwe[HATE] and mwe[0] in count_mwe[NON_HATE]:
#             count[GLOBAL][mwe[1]] += count_mwe[HATE][mwe[0]]
#             count[GLOBAL][mwe[1]] += count_mwe[NON_HATE][mwe[0]]
#         elif mwe[0] in count_mwe[HATE] and mwe[0] not in count_mwe[NON_HATE]:
#             count[HATE][mwe[1]] += count_mwe[HATE][mwe[0]]
#         elif mwe[0] not in count_mwe[HATE] and mwe[0] in count_mwe[NON_HATE]:
#             count[NON_HATE][mwe[1]] += count_mwe[NON_HATE][mwe[0]]
#
#     return count


# def search_mwe_in_tweet_with_discontinuity(tweet, lexicon):
#     import re
#     tweet = tweet.lower()
#     count_mwe = 0
#     for mwe in lexicon:
#         pattern = ''
#         for word in mwe[0].split():
#             if '+' in word:
#                 pattern += "\+" + '\s[\w]+'
#             else:
#                 pattern += word + '\s[\w]+'
#         regex = re.compile(pattern, re.IGNORECASE)
#         if regex.findall(tweet):
#             count_mwe += len(regex.findall(tweet))
#     return count_mwe


def lemmatize_tweet(tweet):

    parsing = nlp(tweet)
    tweet_lemmatize = ''
    for word in parsing:
        tweet_lemmatize += word.lemma_ + ' '
    return tweet_lemmatize


def no_overlaps(mwes):
    if len(mwes) == 1:
        return mwes

    no_overlaps_in_mwes = []
    for x in mwes:
        no_overlaps_bool = True
        for y in mwes:
            if x != y:
                if min(x) >= min(y) >= max(x):
                    no_overlaps_bool = False
                if min(x) >= max(y) >= max(x):
                    no_overlaps_bool = False
        if no_overlaps_bool:
            no_overlaps_in_mwes.append(x)

    return no_overlaps_in_mwes


def tokenize_label_mwe(lexicon):
    vocab_label_mwe = {'<pad>': 0, 'NOMWE': 1}
    vocab_label_mwe_strong = {'<pad>': 0, 'NOMWE': 1}
    for mwe in lexicon:
        if mwe[1] not in vocab_label_mwe:
            vocab_label_mwe[mwe[1]] = len(vocab_label_mwe)
        if mwe[2] not in vocab_label_mwe_strong:
            vocab_label_mwe_strong[mwe[2]] = len(vocab_label_mwe_strong)
    return vocab_label_mwe, vocab_label_mwe_strong


def annotated_corpus(tweets, lexicon, vocab, vocab_strong, max_len_features):
    import numpy as np
    vector_mwe_tweets = []
    vector_mwe_strong_tweets = []

    from tqdm import tqdm
    for tweet in tqdm(tweets):
        try:
            tweet_lemmatized = lemmatize_tweet(tweet).split()
        except:
            tweet_lemmatized = tweet.split()
        vector_tweet = np.zeros(max_len_features)
        vector_tweet_strong = np.zeros(max_len_features)
        for i in range(len(tweet_lemmatized)):
            vector_tweet[i] = 1.0
            vector_tweet_strong[i] = 1.0
        mwes = []
        mwe_label = []
        mwe_label_strong = []
        for mwe in lexicon:
            mwe_split = mwe[0].split()
            for index_lemma in range(len(tweet_lemmatized)):
                if mwe_split[0] == tweet_lemmatized[index_lemma]:
                    mwe_index = []
                    if len(mwe_split) + index_lemma <= len(tweet_lemmatized):
                        for index_mwe in range(len(mwe_split)):
                            if mwe_split[index_mwe] == tweet_lemmatized[index_lemma + index_mwe]:
                                mwe_index.append(index_lemma + index_mwe)
                    if len(mwe_index) == len(mwe_split):
                        mwes.append(mwe_index)
                        mwe_label.append(mwe[1])
                        mwe_label_strong.append(mwe[2])
        mwes = no_overlaps(mwes)
        for mwe in range(len(mwes)):
            for index in mwes[mwe]:
                vector_tweet[index] = vocab[mwe_label[mwe]]  # vocab_label_mwe[mwe_label[mwe]]
                vector_tweet_strong[index] = vocab_strong[mwe_label_strong[mwe]]  # vocab_label_mwe[mwe_label[mwe]]
        vector_mwe_tweets.append(vector_tweet)
        vector_mwe_strong_tweets.append(vector_tweet_strong)
    return np.array(vector_mwe_tweets), np.array(vector_mwe_strong_tweets)


def annotated_corpus_with_discontinuity(tweets, lexicon, vocab, vocab_strong, max_len_features):
    import re
    import numpy as np
    vector_mwe_tweets = []
    vector_mwe_strong_tweets = []
    from tqdm import tqdm
    for tweet in tqdm(tweets):
        try:
            tweet_lemmatized = lemmatize_tweet(reconstruct_sentence(tweet))
        except:
            tweet_lemmatized = tweet
        vector_tweet = np.zeros(max_len_features)
        vector_tweet_strong = np.zeros(max_len_features)

        for i in range(len(tweet_lemmatized.split())):
            vector_tweet[i] = 1.0
            vector_tweet_strong[i] = 1.0
        mwes = []
        mwe_label = []
        mwe_label_strong = []
        for mwe in lexicon:
            pattern_discont = ""
            pattern_cont = ""
            for word in mwe[0].split():
                if '+' in word:
                    pattern_discont += '\+' + '\s' + '[\w]+'
                    pattern_cont += '\+' + '\s'
                else:
                    pattern_discont += word + '\s' + '[\w]+'
                    pattern_cont += word + '\s'

            regex_discont = re.compile(pattern_discont, re.IGNORECASE)
            regex_cont = re.compile(pattern_cont, re.IGNORECASE)
            mwe_candidate = []
            for res in regex_cont.findall(tweet_lemmatized):
                mwe_candidate.append(res)
            for res in regex_discont.findall(tweet_lemmatized):
                mwe_candidate.append(res)
            for mwe_c in mwe_candidate:
                mwe_index = []
                try:
                    for word in mwe_c.split():
                        mwe_index.append(tweet_lemmatized.split().index(word))
                    mwes.append(mwe_index)
                    mwe_label.append(mwe[1])
                    mwe_label_strong.append(mwe[2])
                except:
                    mwe_index=[]
        mwes = no_overlaps(mwes)
        for mwe in range(len(mwes)):
            for index in mwes[mwe]:
                vector_tweet[index] = vocab[mwe_label[mwe]]  # vocab_label_mwe[mwe_label[mwe]]
                vector_tweet_strong[index] = vocab_strong[mwe_label_strong[mwe]]  # vocab_label_mwe[mwe_label[mwe]]
        vector_mwe_tweets.append(vector_tweet)
        vector_mwe_strong_tweets.append(vector_tweet_strong)
    return np.array(vector_mwe_tweets), np.array(vector_mwe_strong_tweets)


def write_vocab_mwe(path, vocab_mwe, vocab_mwe_strong):
    file_vocab_mwe = open(path + '.voc', 'w', encoding='utf-8')
    for key, value in vocab_mwe.items():
        file_vocab_mwe.write(key + '\t' + str(value) + '\n')
    file_vocab_mwe.close()
    file_vocab_mwe_strong = open(path + '.vocstrong', 'w', encoding='utf-8')
    for key, value in vocab_mwe_strong.items():
        file_vocab_mwe_strong.write(key + '\t' + str(value) + '\n')
    file_vocab_mwe_strong.close()


def load_vocab_mwe(path):
    file_vocab_mwe = open(path, 'r', encoding=('utf-8'))
    vocab = {}
    for line in file_vocab_mwe.readlines():
        vocab[line.split()[0]] = int(line.split()[1])
    return vocab


def write_vector(path, vector_mwe):
    import csv
    writer = csv.writer(open(path, 'w', encoding="utf-8"), delimiter="\t")
    for vector in vector_mwe:
        writer.writerow(vector.tolist())


def load_vector(path, size):
    import csv
    import numpy as np
    vectors = []
    reader = csv.reader(open(path, 'r', encoding="utf-8"), delimiter="\t")
    for row in reader:
        assert len(row) == size
        vector = np.zeros(size)
        for v_index in range(len(row)):
            vector[v_index] = float(row[v_index])
        vectors.append(vector)

    return np.array(vectors)


def annotated_mwe(path_lexicon, path_files=[],  discontinuity=False, is_founta=False):
    lexicon = read_lexicon(path_lexicon)
    vocab_mwe, vocab_mwe_strong = tokenize_label_mwe(lexicon)
    # Write vocab, vocab_strong
    write_vocab_mwe(path_lexicon.split(".txt")[0], vocab_mwe, vocab_mwe_strong)

    from utils import read_hateval
    from utils import read_founta

    for path in path_files:
        # Load tweets
        if is_founta:
            tweets, labels, vocab = read_founta(path)
        else:
            tweets, label, vocab = read_hateval(path)
        # Vectorize tweets
        if discontinuity:
            vector_mwe, vector_strong_mwe_features = annotated_corpus_with_discontinuity(
                tweets=tweets, lexicon=lexicon, vocab=vocab_mwe,
                vocab_strong=vocab_mwe_strong,
                max_len_features=280)
        else:
            vector_mwe, vector_strong_mwe_features = annotated_corpus(
                tweets=tweets, lexicon=lexicon, vocab=vocab_mwe,
                vocab_strong=vocab_mwe_strong,
                max_len_features=280)

        # Write vectors
        write_vector(path.split('.csv')[0] + ".mwe." + path_lexicon.split("/")[-1].split(".txt")[0], vector_mwe)
        write_vector(path.split('.csv')[0] + ".mwestrong." + path_lexicon.split("/")[-1].split(".txt")[0],
                     vector_strong_mwe_features)


def annotated_only_mwe_features(path_lexicon, path_files, is_founta):
    import csv
    lexicon = read_lexicon(path_lexicon)
    vocab_mwe, vocab_mwe_strong = tokenize_label_mwe(lexicon)
    # Write vocab, vocab_strong
    write_vocab_mwe(path_lexicon.split(".txt")[0], vocab_mwe, vocab_mwe_strong)

    from utils import read_hateval
    from utils import read_founta
    for path in path_files:
        # Load tweets
        if is_founta:
            tweets, labels, vocab = read_founta(path)
        else:
            tweets, label, vocab = read_hateval(path)
        mwe_features = only_mwe_annotation(tweets=tweets, lexicon=lexicon)

        writer = csv.writer(
            open(path.split('.csv')[0] + ".mwe." + path_lexicon.split("/")[-1].split(".mwelemmas")[0], 'w',
                 encoding='UTF-8'), delimiter="\t")
        for mwe in mwe_features:
            writer.writerow(mwe)


def only_mwe_annotation(tweets, lexicon):
    mwe_features_tweets = []
    from tqdm import tqdm
    for tweet in tqdm(tweets):
        try:
            tweet_lemmatized = lemmatize_tweet(reconstruct_sentence(tweet)).split()
        except:
            tweet_lemmatized = tweet
        mwes = []
        mwe_label = []
        mwe_label_strong = []
        for mwe in lexicon:
            mwe_split = mwe[0].split()
            for index_lemma in range(len(tweet_lemmatized)):
                if mwe_split[0] == tweet_lemmatized[index_lemma]:
                    mwe_index = []
                    if len(mwe_split) + index_lemma < len(tweet_lemmatized) - 1:
                        for index_mwe in range(len(mwe_split)):
                            if mwe_split[index_mwe] == tweet_lemmatized[index_lemma + index_mwe]:
                                mwe_index.append(index_lemma + index_mwe)
                    if len(mwe_index) == len(mwe_split):
                        mwes.append(mwe_index)
                        mwe_label.append(mwe[1])
                        mwe_label_strong.append(mwe[2])
        mwes = no_overlaps(mwes)
        mwe_feat = []
        mwes = sorted(mwes)
        for mwe in range(len(mwes)):
            for index in mwes[mwe]:
                mwe_feat.append(tweet_lemmatized[index])

        mwe_features_tweets.append(mwe_feat)
    return mwe_features_tweets


def read_mwes(path, vocab_mwe, size):
    import csv
    import numpy as np
    reader = csv.reader(open(path, 'r', encoding='UTF-8'), delimiter='\t')
    mwe_features = []
    for row in reader:
        vector = []
        mwes = []
        for index_word in range(len(row)):
            if row[index_word] != '':
                mwes.append(row[index_word])
        for pad in range(size - len(mwes)):
            vector.append(vocab_mwe['<pad>'])
        for mwe in mwes:
            vector.append(vocab_mwe[mwe])

        mwe_features.append(np.array(vector))

    return np.array(mwe_features)


if __name__ == '__main__':
    # MWE = 'smwes'
    # VMWE = 'wmwes'
    import argparse

    parser = argparse.ArgumentParser(description="""
                System to count multi word expressions in tweets.
                """)
    parser.add_argument("--path_files", dest="path_files",
                        required=False, type=str, nargs='+',
                        help="""
                            File at HatEval (Basile et al., 2019) and Founta (Founta et al., 2018) format. 
                            """)
    parser.add_argument("--path_lexicon", dest="path_lexicon",
                        required=True, type=str,
                        help="""
                        Lexicon path. 
                        """)
    parser.add_argument("--discontinuity", dest="discontinuity",
                        required=False, type=bool, nargs='?', const=True,
                        help="""
                        Option to use annotate the corpus with the possible discontinuity of the MWE.
                        """)
    parser.add_argument("--founta", dest="founta",
                        required=False, type=bool, nargs='?', const=True,
                        help="""
                        If you are using Founta et al. (2018) corpus. 
                        """)
    parser.add_argument("--word_embeddings_feature", dest="word_embeddings_features",
                        required=False, type=bool, nargs='?', const=True,
                        help="""
                        Option to extract words which composed a MWE. 
                        """)

    args = parser.parse_args()

    if args.word_embeddings_features:
        annotated_only_mwe_features(args.path_lexicon, args.path_files, args.founta)
    else:
        annotated_mwe(args.path_lexicon, args.path_files,  args.discontinuity, args.founta)
