# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
from io import open
from sklearn.model_selection import KFold, train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from interp2adver import GenerateAdvSamplesLASSOL2


def example_rf(X, y):
    clf = RF(n_estimators=10, min_samples_leaf=5)   # just an example, not tuned
    clf.fit(X, y)
    return clf


def main():
    parser = ArgumentParser("interp2adver",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')
    parser.add_argument('--input', default='example_data/data_small.npy',
                      help='Input data')    # A small toy dataset
    parser.add_argument('--seed-ratio', default=0.05, type=float,
                      help='Ratio of seeds among overall data')
    parser.add_argument('--dist-ratio-min', default=0.1, type=float,
                      help='Ratio of seeds among overall data')
    parser.add_argument('--dist-ratio-max', default=0.5, type=float,
                        help='Ratio of seeds among overall data')
    parser.add_argument('--dist-ratio-num', default=5, type=float,
                        help='Ratio of seeds among overall data')

    args = parser.parse_args()
    seed_ratio = args.seed_ratio
    dist_ratio_arr = np.linspace(args.dist_ratio_min, args.dist_ratio_max, num=args.dist_ratio_num)

    # Get data
    data_mat = np.load(args.input)
    X = preprocessing.scale(data_mat[:, 1:])
    X -= np.amin(X)
    y = data_mat[:, 0]

    # Data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    print "Get training and testing data."

    # Train the classifier
    clf = example_rf(X_train, y_train)
    print "Prediction performance: "
    print precision_recall_fscore_support(y_test, clf.predict(X_test), average='binary')
    print '\n'

    # Generate adversarial samples
    adv_insts_new = GenerateAdvSamplesLASSOL2(X_train, y_train, clf, dist_ratio_arr, seed_ratio)

    # Test
    for adv_x in [adv_insts_new[d] for d in range(len(dist_ratio_arr))]:
        adv_insts_train, adv_insts_test, adv_y_train, adv_y_test = \
            train_test_split(adv_x, np.ones(adv_x.shape[0]), test_size=0.25, random_state=1)

        clf_new =  example_rf(np.vstack((X_train,adv_insts_train)), np.hstack((y_train, adv_y_train)))

        print "Test Result of the new detector for old test data: \n", \
            precision_recall_fscore_support(y_test, clf_new.predict(X_test), average='binary')
        print "Test Result of the new detector for adversarial test data:\n ", \
            precision_recall_fscore_support(adv_y_test, clf_new.predict(adv_insts_test), average='binary')
        print "Test Result of the old detector for adversarial test data:\n ", \
            precision_recall_fscore_support(adv_y_test, clf.predict(adv_insts_test), average='binary')
        print '\n'


if __name__ == "__main__":
  sys.exit(main())
