"""
enviroment of IMDB classify
"""

import random
import numpy as np
import os
import re
POSITIVE_SIZE = 12500
NEGTIVE_SIZE = 1250
RATION = NEGTIVE_SIZE/POSITIVE_SIZE

class environment():
    def __init__(self):
        index1 = [i for i in range(12500)]
        index2 = [i for i in range(12500,25000)]
        #抽样少量样本
        index2 = random.sample(index2,NEGTIVE_SIZE)
        index_train = index1 + index2
        index_test = list(range(25000))
        #读标号
        test_data = self.load_data('C:/Users/13357/Desktop/aclImdb', flag = 'test')
        train_data = self.load_data('C:/Users/13357/Desktop/aclImdb')
        train_data = [train_data[i] for i in index_train]
        #test_data = [test_data[i] for i in index_test]
        #加载文件
        sentence_code_1 = np.load('E:\IMDB\data\sentence_code_1.npy', allow_pickle=True)
        sentence_code_1 = sentence_code_1.tolist()
        sentence_code_1 = [sentence_code_1[i] for i in index_train]
        sentence_code_2 = np.load('E:\IMDB\data\sentence_code_2.npy', allow_pickle=True)
        sentence_code_2 = sentence_code_2.tolist()
        sentence_code_2 = [sentence_code_2[i] for i in index_test]
        vocabulary_vectors = np.load('E:\IMDB\data/vocabulary_vectors_1.npy', allow_pickle=True)
        vocabulary_vectors = vocabulary_vectors.tolist()
        #转化为词向量
        for i in range(POSITIVE_SIZE + NEGTIVE_SIZE):
            for j in range(250):
                sentence_code_1[i][j] = vocabulary_vectors[sentence_code_1[i][j]]
        for i in range(25000):
            for j in range(250):
                sentence_code_2[i][j] = vocabulary_vectors[sentence_code_2[i][j]]
        self.train_list = np.array(sentence_code_1) #样本数*250*50
        self.train_label = np.array([train_data[i][1] for i in range(POSITIVE_SIZE + NEGTIVE_SIZE)]) #样本数*1
        self.test_list = np.array(sentence_code_2)
        self.test_label = np.array([test_data[i][1] for i in range(25000)])
        self.s = random.randint(0,POSITIVE_SIZE + NEGTIVE_SIZE - 1)

    def reset(self):
        self.s = random.randint(0,POSITIVE_SIZE + NEGTIVE_SIZE - 1)
        return self.train_list[self.s]

    def step(self,action):
        if self.s > POSITIVE_SIZE:
            if action == self.train_label[self.s]: 
                self.s = random.randint(0,POSITIVE_SIZE + NEGTIVE_SIZE - 1)
                return self.train_list[self.s], 1, False
            else : return self.train_list[self.s], 0, True
        else:
            if action == self.train_label[self.s]: 
                self.s = random.randint(0,POSITIVE_SIZE + NEGTIVE_SIZE - 1)
                return self.train_list[self.s], RATION, False
            else : 
                self.s = random.randint(0,POSITIVE_SIZE + NEGTIVE_SIZE - 1)
                return self.train_list[self.s], -RATION, False

    def load_data(self, path, flag='train'):
        labels = ['pos', 'neg']
        data = []
        for label in labels:
            files = os.listdir(os.path.join(path, flag, label))
            # 去除标点符号
            r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
            for file in files:
                with open(os.path.join(path, flag, label, file), 'r', encoding='utf8') as rf:
                    temp = rf.read().replace('\n', '')
                    temp = temp.replace('<br /><br />', ' ')
                    temp = re.sub(r, '', temp)
                    temp = temp.split(' ')
                    temp = [temp[i].lower() for i in range(len(temp)) if temp[i] != '']
                    if label == 'pos':
                        data.append([temp, 1])
                    elif label == 'neg':
                        data.append([temp, 0])
        return data