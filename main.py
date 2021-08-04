# -*- coding: utf-8 -*-
r"""
    Author: Nicolas Zampieri
    Date: August 26, 2020
"""

# IMPORT
import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from utils import one_hot_weight, read_founta, read_hateval, prediction_to_class, \
    write_prediction_hateval, write_prediction_founta, plot_history_acc, plot_history_loss, read_davidson, \
    prediction_to_class_softmax_founta, prediction_to_class_softmax_davidson


def use_model(is_founta, is_davidson):
    try:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

    inputs = Input(512, name='use_embed')
    dense_use = Dense(256, activation="relu")(inputs)

    if is_founta or is_davidson:
        output = Dense(3, activation='softmax')(dense_use)
    else:
        output = Dense(1, activation='sigmoid')(dense_use)
    model = Model(inputs=inputs, outputs=output)
    if is_founta or is_davidson:
        model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
    else:
        model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])
    model.summary()
    return model


def use_mwecat_cnn_embeddings_w2v(mwe_categories_features, mwe_embeddings_features, sentence_len, length_w2v_embeddings, is_founta, is_davidson):
    try:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    inputs = [Input(512, name='use_embed'), # USE features
              Input(shape=(sentence_len,), name=mwe_categories_features["name"]), # MWE categories features
              Input(shape=(length_w2v_embeddings,), name=mwe_embeddings_features["name"])] # MWE embeddings features

    # USE branch
    dense_use = Dense(256, activation='relu')(inputs[0])

    # MWE categories branch
    embeddings_one_hot = Embedding(input_dim=mwe_categories_features['input_dim'], output_dim=mwe_categories_features['output_dim'],
                                   input_length=sentence_len,
                                   weights=[mwe_categories_features['weights']],
                                   embeddings_initializer=mwe_categories_features['initializer'],
                                   trainable=mwe_categories_features['trainable'], mask_zero=mwe_categories_features['mask_zero'])(
        inputs[1])
    cnn_one_hot = Conv1D(32, 3, activation='relu')(embeddings_one_hot)
    cnn_one_hot = MaxPooling1D()(cnn_one_hot)
    cnn_one_hot = Conv1D(16, 3, activation='relu')(cnn_one_hot)
    cnn_one_hot = MaxPooling1D()(cnn_one_hot)
    cnn_one_hot = Conv1D(8, 3, activation='relu')(cnn_one_hot)
    cnn_one_hot = MaxPooling1D()(cnn_one_hot)
    output_cnn_one_hot = Flatten()(cnn_one_hot)

    # MWE embeddings branch
    embeddings_embeddings = Embedding(input_dim=mwe_embeddings_features['input_dim'],
                                      output_dim=mwe_embeddings_features['output_dim'],
                                      input_length=sentence_len,
                                      weights=[mwe_embeddings_features['weights']],
                                      embeddings_initializer=mwe_embeddings_features['initializer'],
                                      trainable=mwe_embeddings_features['trainable'],
                                      mask_zero=mwe_embeddings_features['mask_zero'])(inputs[2])
    lstm_mwe_embed = LSTM(192)(embeddings_embeddings)

    concat = Concatenate()([dense_use, output_cnn_one_hot, lstm_mwe_embed])

    dense = Dense(256, activation="relu")(concat)
    if is_founta or is_davidson:
        output = Dense(3, activation='softmax')(dense)
    else:
        output = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=inputs, outputs=output)
    if is_founta or is_davidson:
        model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
    else:
        model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])
    model.summary()
    return model


def use_mwecat_cnn_embeddings_bert_lstm(mwe_categories_features, sentence_len, length_bert_embeddings, is_founta, is_davidson):
    try:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    inputs = [Input(512, name='use_embed'),
              Input(shape=(sentence_len,), name=mwe_categories_features["name"]),
              Input(shape=(length_bert_embeddings, 768), name='bert_mwe_embeddings')]

    # USE branch
    dense_use = Dense(256, activation='relu')(inputs[0])

    # MWE categories branch
    embeddings_one_hot = Embedding(input_dim=mwe_categories_features['input_dim'], output_dim=mwe_categories_features['output_dim'],
                                   input_length=sentence_len,
                                   weights=[mwe_categories_features['weights']],
                                   embeddings_initializer=mwe_categories_features['initializer'],
                                   trainable=mwe_categories_features['trainable'], mask_zero=mwe_categories_features['mask_zero'])(
        inputs[1])
    cnn_one_hot = Conv1D(32, 3, activation='relu')(embeddings_one_hot)
    cnn_one_hot = MaxPooling1D()(cnn_one_hot)
    cnn_one_hot = Conv1D(16, 3, activation='relu')(cnn_one_hot)
    cnn_one_hot = MaxPooling1D()(cnn_one_hot)
    cnn_one_hot = Conv1D(8, 3, activation='relu')(cnn_one_hot)
    cnn_one_hot = MaxPooling1D()(cnn_one_hot)
    output_cnn_one_hot = Flatten()(cnn_one_hot)
    #lstm_one_hot = LSTM(128, return_sequences=False)(embeddings_one_hot)


    # MWE embeddings branch
    lstm_bert = LSTM(192)(inputs[-1])

    # Concatenation
    concat = Concatenate()([dense_use, output_cnn_one_hot, lstm_bert])
    dense = Dense(256, activation="relu")(concat)
    if is_founta or is_davidson:
        output = Dense(3, activation='softmax')(dense)
    else:
        output = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=inputs, outputs=output)
    if is_founta or is_davidson:
        model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
    else:
        model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])
    model.summary()
    return model


def use_mwecat_lstm_embeddings_bert_lstm(mwe_categories_features, sentence_len, length_bert_embeddings, is_founta, is_davidson):
    try:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    inputs = [Input(512, name='use_embed'),
              Input(shape=(sentence_len,), name=mwe_categories_features["name"]),
              Input(shape=(length_bert_embeddings, 768), name='bert_mwe_embeddings')]

    # USE branch
    dense_use = Dense(256, activation='relu')(inputs[0])

    # MWE categories branch
    embeddings_one_hot = Embedding(input_dim=mwe_categories_features['input_dim'], output_dim=mwe_categories_features['output_dim'],
                                   input_length=sentence_len,
                                   weights=[mwe_categories_features['weights']],
                                   embeddings_initializer=mwe_categories_features['initializer'],
                                   trainable=mwe_categories_features['trainable'], mask_zero=mwe_categories_features['mask_zero'])(
        inputs[1])

    lstm_one_hot = LSTM(128, return_sequences=False)(embeddings_one_hot)

    # MWE embeddings branch
    lstm_bert = LSTM(192)(inputs[-1])

    # Concatenation
    concat = Concatenate()([dense_use, lstm_one_hot, lstm_bert])
    dense = Dense(256, activation="relu")(concat)
    if is_founta or is_davidson:
        output = Dense(3, activation='softmax')(dense)
    else:
        output = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=inputs, outputs=output)
    if is_founta or is_davidson:
        model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
    else:
        model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_model", dest="model", type=str, required=True,
                        help="Path for model without extension.")
    parser.add_argument("--file_train", dest="train", type=str, required=False,
                        help="Path for train corpus.")
    parser.add_argument("--file_dev", dest="dev", type=str, required=False,
                        help="Path for dev corpus (If available). This corpus is using to validation_data")
    parser.add_argument("--file_test", dest="test", type=str, required=False,
                        help="Path for test corpus. To test the system")
    parser.add_argument("--file_prediction", dest="prediction", type=str, required=True,
                        help="Path of prediction file.")
    parser.add_argument("--file_embeddings", dest="embeddings", type=str, required=False,
                        help="Path of binary Word2vec/FastText model.")
    parser.add_argument("--patience", required=False, metavar="patience",
                        dest="patience", type=int, default=5,
                        help="""
                        Option to choice patience for the early stopping.
                        By default, it is 3.
                        """)
    parser.add_argument("--epochs", required=False, metavar="epochs",
                        dest="epochs", type=int, default=100,
                        help="""
                        Option to choice number of epochs.
                        By default, it is 10.
                        """)
    parser.add_argument("--mwe_one_hot", dest="mwe_one_hot", type=str, required=False,
                        help="""Option to use local multi-word expressions features.
                             Take in argument the specific lexicon of Streusle.
                             Refer to --help to see how argument used.
                             """)
    parser.add_argument("--mwe_embeddings", dest="mwe_embeddings", type=str, required=False,
                        help="""Option to use local multi-word expressions features.
                             Take in argument the specific lexicon of Streusle.
                             Refer to --help to see how argument used.
                             """)
    parser.add_argument("--mwe_embeddings_bert", dest="mwe_embeddings_bert", type=str, required=False,
                        help="""Option to use local multi-word expressions features with bert embeddings.
                             Take in argument the extension of train/dev/test for bert embeddings.
                             Refer to --help to see how argument used.
                             """)
    parser.add_argument("--len_embeddings", dest="len_embeddings_bert", type=int, required=False, default=15,
                        help="""Option to set length of local multi-word expressions features with bert embeddings.
                             Default is set to 15.
                             Refer to --help to see how argument used.
                             """)
    parser.add_argument("--founta", dest="founta", type=bool, required=False, nargs='?',
                        const=True,
                        help="""Option to use founta dataset.
                             Refer to --help to see how argument used.
                             """)
    parser.add_argument("--davidson", dest="davidson", type=bool, required=False, nargs='?',
                        const=True,
                        help="""Option to use davidson dataset.
                             Refer to --help to see how argument used.
                             """)
    parser.add_argument("--mwe_features", dest="mwe_features", type=bool, required=False, nargs='?',
                        const=True,
                        help="""Option to use mwe features one_hot/embeddings.
                             Refer to --help to see how argument used.
                             """)
    parser.add_argument("--mwe_cat_lstm", dest="mwe_cat_lstm", type=bool, required=False, nargs='?',
                        const=True,
                        help="""Option to use lstm layer instead of CNN layer for mwe cat features.
                             Refer to --help to see how argument used.
                             """)
    parser.add_argument("--max_sentence_length", dest="max_sentence_length", type=int, required=False, default=280,
                        help="To initialise length max of sentence.")
    parser.add_argument("--batch_size", required=False, metavar="batch_size",
                        dest="batch_size", type=int, default=100,
                        help="""
                        Option to choice number of sentence per batch.
                        By default, it is 100.
                        """)
    parser.add_argument("--save_training", dest="save_training", type=bool, required=False, nargs='?',
                        const=True,
                        help="""Option to save graphics of loss and acc.
                             """)
    args = parser.parse_args()
    from extract_use_embeddings import load_embeddings

    if args.train:
        print("Load train file")
        if args.founta:
            X_train_no_tokenize, Y_train, vocab_train = read_founta(args.train)
        elif args.davidson:
            X_train_no_tokenize, Y_train, vocab_train = read_davidson(args.train)
        else:
            X_train_no_tokenize, Y_train, vocab_train = read_hateval(args.train)

        print("Load dev file")
        if args.founta:
            X_dev_no_tokenize, Y_dev, vocab_dev = read_founta(args.dev)
        elif args.davidson:
            X_dev_no_tokenize, Y_dev, vocab_dev = read_davidson(args.dev)
        else:
            X_dev_no_tokenize, Y_dev, vocab_dev = read_hateval(args.dev)

        # FEATURES
        X_train = load_embeddings(args.train.split(".csv")[0] + ".usembed", size=512)
        X_dev = load_embeddings(args.dev.split(".csv")[0] + ".usembed", size=512)

        X_TRAIN = [np.asarray(X_train).astype(np.float32)]
        X_DEV = [np.asarray(X_dev).astype(np.float32)]
        features_spec = []
        features_input = []
        if args.mwe_features:
            import mwe_features
            if args.mwe_one_hot:
                vocab_mwe = mwe_features.load_vocab_mwe(path=args.mwe_one_hot)
                train_mwe_features = mwe_features.load_vector(
                    args.train.split(".csv")[0] + '.mwe.' + args.mwe_one_hot.split("/")[-1].split(".voc")[0],
                    size=args.max_sentence_length)
                dev_mwe_features = mwe_features.load_vector(
                    args.dev.split(".csv")[0] + '.mwe.' + args.mwe_one_hot.split("/")[-1].split(".voc")[0],
                    size=args.max_sentence_length)
                X_TRAIN.append(train_mwe_features)
                X_DEV.append(dev_mwe_features)

                mwe_categories_spec = {'name': 'MWE_One_hot',
                                      'output_dim': len(vocab_mwe),
                                      'weights': one_hot_weight(vocab_mwe),
                                      'trainable': False,
                                      'input_dim': len(vocab_mwe),
                                      'initializer': 'uniform',
                                      'mask_zero': False}
            if args.mwe_embeddings_bert:
                from extract_bert_embeddings_for_mwe_features import load_mwe_features_bert_embeddings
                train_mwe_features = load_mwe_features_bert_embeddings(args.train.split('.csv')[0] + '.' + args.mwe_embeddings_bert + '.embed')
                dev_mwe_features = load_mwe_features_bert_embeddings(args.dev.split('.csv')[0] + '.' + args.mwe_embeddings_bert + '.embed')
                X_TRAIN.append(train_mwe_features)
                X_DEV.append(dev_mwe_features)
            elif args.embeddings and args.mwe_embeddings:
                lexicon_mwe = mwe_features.read_lexicon(args.mwe_embeddings)
                vocab_mwe = {'<pad>': 0}
                for mwe in lexicon_mwe:
                    for word in mwe[0].split():
                        if word not in vocab_mwe:
                            vocab_mwe[word] = len(vocab_mwe)
                from gensim.models import KeyedVectors, fasttext
                import numpy as np

                embed = None
                matrix = []

                try:
                    try:
                        embed = KeyedVectors.load_word2vec_format(args.embeddings, binary=True,
                                                                  unicode_errors='ignore')
                    except:
                        embed = fasttext.load_facebook_vectors(args.embeddings)
                except:
                    print(
                        "Error to load embedding " + args.embeddings + ". Please, check if the file is Word2vec or FastText model.",
                        file=sys.stderr)
                    exit(-1)
                for word, key in vocab_mwe.items():
                    if word in embed.vocab:
                        matrix.append(embed[word])
                    else:
                        matrix.append(np.zeros(400))
                print(len(vocab_mwe), len(matrix))
                train_mwe_features = mwe_features.read_mwes(
                    args.train.split(".csv")[0] + '.mwe.' + args.mwe_embeddings.split("/")[-1], vocab_mwe,
                    size=args.len_embeddings)
                dev_mwe_features = mwe_features.read_mwes(
                    args.dev.split(".csv")[0] + '.mwe.' + args.mwe_embeddings.split("/")[-1], vocab_mwe,
                    size=args.len_embeddings)
                X_TRAIN.append(train_mwe_features)
                X_DEV.append(dev_mwe_features)

                mwe_embeddings_spec = {'name': 'MWEs_embeddings',
                                      'output_dim': 400,
                                      'weights': np.array(matrix),
                                      'trainable': False,
                                      'input_dim': len(vocab_mwe),
                                      'initializer': 'uniform',
                                      'mask_zero': False}
            else:
                print("Error, you need to give mwe embeddings features. You need to use --mwe_embeddings_bert or --file_embeddings with --mwe_embeddings.", file=sys.stderr)
                exit(1000)

        # Models
        if args.mwe_embeddings_bert:
            if args.mwe_cat_lstm:
                model = use_mwecat_lstm_embeddings_bert_lstm(mwe_categories_spec, args.max_sentence_length, args.len_embeddings, args.founta, args.davidson)
            else:
                model = use_mwecat_cnn_embeddings_bert_lstm(mwe_categories_spec, args.max_sentence_length, args.len_embeddings, args.founta, args.davidson)
        elif args.mwe_embeddings:
            model = use_mwecat_cnn_embeddings_w2v(mwe_categories_spec, mwe_embeddings_spec, args.max_sentence_length, args.len_embeddings, args.founta, args.davidson)
        else:
            model = use_model(args.founta, args.davidson)

        # If model use only USE features
        if not args.mwe_features:
            X_TRAIN = X_train
            X_DEV = X_dev

        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
        from tensorflow.keras.utils import to_categorical

        if args.founta or args.davidson:
            Y_train = to_categorical(Y_train)
            Y_dev = to_categorical(Y_dev)
        else:
            Y_train = np.asarray(Y_train).astype(np.int)
            Y_dev = np.asarray(Y_dev).astype(np.int)
        print("Training...")
        checkpoint = ModelCheckpoint(args.model + '.h5', monitor='val_loss', verbose=1,
                                     save_best_only=True,
                                     mode='min')
        earlyStopping = EarlyStopping(monitor="val_loss", patience=args.patience, verbose=1, mode="min")
        callbacks_list = [checkpoint, earlyStopping]
        history = model.fit(X_TRAIN, Y_train, batch_size=args.batch_size,
                            validation_data=(X_DEV, Y_dev), epochs=args.epochs,
                            callbacks=callbacks_list)

        if args.save_training:
            plot_history_acc(history, args.model + '-acc.png')
            plot_history_loss(history, args.model + '-loss.png')

    # TESTING
    if args.test is not None:
        print("Load test file")
        if args.founta:
            X_test_no_tokenize, Y_test, vocab_test = read_founta(args.test)
        elif args.davidson:
            X_test_no_tokenize, Y_test, vocab_test = read_davidson(args.test)
        else:
            X_test_no_tokenize, Y_test, vocab_test = read_hateval(args.test)
        print("Load model and vocab")
        try:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass
        model = load_model(args.model + ".h5")
        model.summary()

        # FEATURES
        X_test = load_embeddings(args.test.split(".csv")[0] + ".usembed", size=512)
        X_TEST = [X_test]

        if args.mwe_features:
            import mwe_features

            if args.mwe_one_hot:
                mwe_features_one_hot = mwe_features.load_vector(
                    args.test.split(".csv")[0] + '.mwe.' + args.mwe_one_hot.split("/")[-1].split(".voc")[0],
                    size=args.max_sentence_length)
                X_TEST.append(mwe_features_one_hot)
            if args.mwe_embeddings_bert:
                from extract_bert_embeddings_for_mwe_features import load_mwe_features_bert_embeddings
                test_mwe_features = load_mwe_features_bert_embeddings(args.test.split('.csv')[0] + '.' + args.mwe_embeddings_bert + '.embed')
                X_TEST.append(test_mwe_features)
            elif args.embeddings and args.mwe_embeddings:
                lexicon_mwe = mwe_features.read_lexicon(args.mwe_embeddings)

                vocab_mwe = {'<pad>': 0}
                for mwe in lexicon_mwe:
                    for word in mwe[0].split():
                        if word not in vocab_mwe:
                            vocab_mwe[word] = len(vocab_mwe)
                mwe_features_embeddings = mwe_features.read_mwes(
                    args.test.split(".csv")[0] + '.mwe.' + args.mwe_embeddings.split("/")[-1], vocab_mwe,
                    size=args.max_sentence_length)
                X_TEST.append(mwe_features_embeddings)
            else:
                print("Error, you need to give mwe embeddings features. You need to use --mwe_embeddings_bert or --file_embeddings with --mwe_embeddings.", file=sys.stderr)
                exit(1000)
        # If model use only USE features
        if not args.mwe_features:
            X_TEST = X_test
        Y_pred = model.predict(X_TEST)
        if args.founta:
            write_prediction_founta(args.test, args.prediction, prediction_to_class_softmax_founta(Y_pred))
        elif args.davidson:
            write_prediction_founta(args.test, args.prediction, prediction_to_class_softmax_davidson(Y_pred))
        else:
            write_prediction_hateval(args.test, args.prediction, prediction_to_class(Y_pred))
