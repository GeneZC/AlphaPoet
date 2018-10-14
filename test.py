import numpy as np
import tensorflow as tf
import os
import pickle
import random
from generator import Generator
from mobilenet import MobileNet
from PIL import Image

EMB_DIM = 300 # embedding dimension
HIDDEN_DIM = 300 # hidden state dimension of lstm cell
SEQ_LENGTH = 12 # sequence length
START_TOKEN = 0
SEED = 88
BATCH_SIZE = 1

DICO= 'tang.txt'
DICO_PKL = 'dict.pkl'
IMAGE = 'test/3.jpg'

def create_dico(filename):
    dico = {}
    dico['unk'] = 1000000
    dico['sos'] = 1000001
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.strip():
                if word != ' ':
                    if word not in dico:
                        dico[word] = 1
                    else:
                        dico[word] += 1
    sorted_items = sorted(dico.items(), key=lambda x: (x[1], x[0]), reverse=True)
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    if os.path.exists(DICO_PKL):
        with open(DICO_PKL, 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
    else:
        word_to_id, id_to_word = create_dico(DICO)
        with open(DICO_PKL, 'wb') as f:
            pickle.dump([word_to_id, id_to_word], f)

    vocab_size = len(word_to_id)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, True)
    mobilenet = MobileNet(BATCH_SIZE)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # generator.load_weight()
    mobilenet.load_pretrained_weights(sess)
    sess.run(tf.global_variables_initializer())  

    im = Image.open(IMAGE).convert('RGB')
    im = im.resize((224, 224))
    im = np.array(im)
    im = np.expand_dims(im, 0)
    feed_dict = {
                    mobilenet.X: im,
                    mobilenet.is_training: False 
                }
    hidden_batch = sess.run(mobilenet.y_output, feed_dict=feed_dict)
    samples = generator.generate(sess, hidden_batch)
    y = samples.tolist()
    for k, sam in enumerate(y):
        sa = [id_to_word[i] for i in sam]
        sa = ''.join(sa)
        print(sa)

if __name__ == '__main__':
    main()