import numpy as np
from PIL import Image
import os
import random

class Gen_Data_loader():
    def __init__(self, batch_size, word_to_id):
        self.batch_size = batch_size
        self.word_to_id = word_to_id
        self.token_stream = []

    def create_batches(self, data_file, image_dir):
        self.inp_stream = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                inp = [self.word_to_id[x] if x in self.word_to_id else self.word_to_id['unk'] for x in line]
                self.inp_stream.append(inp)
        self.image_stream = []
        image_num = len(os.listdir(image_dir))
        # self.image_stream = random.sample(self.image_stream, image_num)
        for image_idx in range(image_num):
            im = Image.open(image_dir+str(image_idx+1)+'.jpg').convert('RGB')
            im = im.resize((224, 224))
            im = np.array(im)
            self.image_stream.append(im)
        self.num_batch = int(len(self.inp_stream) / self.batch_size)
        self.inp_stream = self.inp_stream[:self.num_batch * self.batch_size]
        self.inp_batch = np.split(np.array(self.inp_stream), self.num_batch, 0)
        self.image_stream = self.image_stream[:self.num_batch * self.batch_size]
        self.image_batch = np.split(np.array(self.image_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self, shuffle=False):
        if shuffle:
            inp = self.inp_batch[random.randint(0, self.num_batch-1)]
        else:
            inp = self.inp_batch[self.pointer]
        image = self.image_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return inp, image

    def reset_pointer(self):
        self.pointer = 0

class Dis_Data_loader():
    def __init__(self, batch_size, word_to_id):
        self.batch_size = batch_size
        self.word_to_id = word_to_id
        self.sentences = np.array([])
        self.labels = np.array([])

    def create_batches(self, datas, samples):
        positive_examples = datas
        negative_examples = samples
        self.sentences = np.array(positive_examples+negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
