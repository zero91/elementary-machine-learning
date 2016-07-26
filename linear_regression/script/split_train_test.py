from __future__ import division
import sys
import os
import random
import numpy as np
import pandas as pd

if __name__ == "__main__":
    #print np.bincount(pd.read_csv("../data_in/train.csv", sep=';')['quality'].values) / sum(np.bincount(pd.read_csv("../data_in/train.csv", sep=';')['quality'].values))
    #print np.bincount(pd.read_csv("../data_in/test.csv", sep=';')['quality'].values) / sum(np.bincount(pd.read_csv("../data_in/test.csv", sep=';')['quality'].values))
    #print np.bincount(pd.read_csv("../data_in/validate.csv", sep=';')['quality'].values) / sum(np.bincount(pd.read_csv("../data_in/validate.csv", sep=';')['quality'].values))
    ftrain = open("../data_in/train.csv", 'w')
    fval = open("../data_in/validate.csv", 'w')
    ftest = open("../data_in/test.csv", 'w')
    with open("../data_in/winequality-white.csv", 'r') as fin:
        head = fin.readline()
        ftrain.write(head)
        fval.write(head)
        ftest.write(head)
        for line in fin.xreadlines():
            random_val = random.random()
            if random_val < 0.7:
                ftrain.write(line)
            elif random_val < 0.9:
                fval.write(line)
            else:
                ftest.write(line)
