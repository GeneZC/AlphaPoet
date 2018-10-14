import numpy as np
import tensorflow as tf
import random
import pickle
import os
from dataloader import Gen_Data_loader, Dis_Data_loader
from generator import Generator
from discriminator import BLEUCNN
from rollout import ROLLOUT
from mobilenet import MobileNet

EMB_DIM = 300 # embedding dimension
HIDDEN_DIM = 300 # hidden state dimension of lstm cell
SEQ_LENGTH = 12 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 1 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

RL_EPOCH_NUM = 30
DICO_PKL = 'dict.pkl'
DICO= 'tang.txt'
CORPUS = 'data/temp.txt'
IMAGE = 'data/temp/'

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

def create_data(filename, word_to_id):
    inp_stream = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            inp_stream.append([word_to_id[line[word_idx]] for word_idx in range(SEQ_LENGTH)])
    return inp_stream

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    if os.path.exists(DICO_PKL):
        with open(DICO_PKL, 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
    else:
        word_to_id, id_to_word = create_dico(DICO)
        with open(DICO_PKL, 'wb') as f:
            pickle.dump([word_to_id, id_to_word], f)
    
    gen_data_loader = Gen_Data_loader(BATCH_SIZE, word_to_id)
    dis_data_loader = Dis_Data_loader(BATCH_SIZE, word_to_id)
    vocab_size = len(word_to_id)
    assert START_TOKEN == word_to_id['sos']

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    discriminator = BLEUCNN(SEQ_LENGTH, 2, EMB_DIM, generator)
    mobilenet = MobileNet(BATCH_SIZE)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    mobilenet.load_pretrained_weights(sess)
    sess.run(tf.global_variables_initializer())

    log = open('experiment-log.txt', 'w', encoding='utf-8')
    #  pre-train generator and discriminator
    log.write('pre-training...\n')
    print('Start pre-training discriminator...')
    datas = create_data(DICO, word_to_id)
    gen_data_loader.create_batches(CORPUS, IMAGE)
    samples = []
    for it in range(gen_data_loader.num_batch):
        inp_batch, image_batch = gen_data_loader.next_batch()
        feed_dict = {
                    mobilenet.X: image_batch,
                    mobilenet.is_training: False 
        }
        hidden_batch = sess.run(mobilenet.y_output, feed_dict=feed_dict)
        samples.extend(generator.generate(sess, hidden_batch).tolist())
    dis_data_loader.create_batches(random.sample(datas, 3000), samples)
    for _ in range(PRE_EPOCH_NUM):
        dis_data_loader.reset_pointer()
        for it in range(dis_data_loader.num_batch):
            x_batch, labels = dis_data_loader.next_batch()
            feed = {
                discriminator.input_x: x_batch,
                discriminator.labels: labels,
                discriminator.dropout_keep_prob: 0.75
            }
            _ = sess.run(discriminator.train_op, feed)

    print('Start pre-training generator...')
    for epoch in range(PRE_EPOCH_NUM):
        supervised_g_losses = []
        gen_data_loader.reset_pointer()
        for it in range(gen_data_loader.num_batch):
            inp_batch, image_batch = gen_data_loader.next_batch()
            feed_dict = {
                    mobilenet.X: image_batch,
                    mobilenet.is_training: False 
            }
            hidden_batch = sess.run(mobilenet.y_output, feed_dict=feed_dict)
            _, g_loss = generator.pretrain_step(sess, inp_batch, hidden_batch)
            supervised_g_losses.append(g_loss)
        loss = np.mean(supervised_g_losses)
        if epoch % 5 == 0:
            print('pre-train epoch ', epoch, 'train_loss ', loss)
            buffer = 'epoch:\t'+ str(epoch) + '\ttrain_loss:\t' + str(loss) + '\n'
            log.write(buffer)

    rollout = ROLLOUT(generator, 0.8)

    print('#########################################################################')
    print('Start REINFORCE Training...')
    log.write('REINFORCE training...\n')
    for total_batch in range(RL_EPOCH_NUM):
        gen_data_loader.reset_pointer()
        for it in range(gen_data_loader.num_batch):
            ra = random.randint(0, 1)
            inp_batch, image_batch = gen_data_loader.next_batch(shuffle=ra)
            feed_dict = {
                    mobilenet.X: image_batch,
                    mobilenet.is_training: False 
            }
            hidden_batch = sess.run(mobilenet.y_output, feed_dict=feed_dict)
            samples = generator.generate(sess, hidden_batch)
            rewards = rollout.get_reward(sess, samples, hidden_batch, 16, discriminator)
            feed = {generator.x:inp_batch, generator.rewards: rewards, generator.hiddens:hidden_batch}
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        if total_batch % 5 == 0 or total_batch == RL_EPOCH_NUM - 1:
            mean_rewards = []
            gen_data_loader.reset_pointer()
            for it in range(gen_data_loader.num_batch):
                inp_batch, image_batch = gen_data_loader.next_batch()
                feed_dict = {
                    mobilenet.X: image_batch,
                    mobilenet.is_training: False 
                }
                hidden_batch = sess.run(mobilenet.y_output, feed_dict=feed_dict)
                samples = generator.generate(sess, hidden_batch)
                rewards = rollout.get_reward(sess, samples, hidden_batch, 16, discriminator)
                mean_rewards.append(np.mean(rewards[:,-1]))
            reward = np.mean(mean_rewards)
            buffer = 'epoch:\t' + str(total_batch) + '\treward:\t' + str(reward) + '\n'
            print('total_batch: ', total_batch, 'reward: ', reward)
            log.write(buffer)
            generator.save_weight(sess)

        # Update roll-out parameters
        rollout.update_params()
        discriminator.update_embedding()

        # Train the discriminator
        samples = []
        for it in range(gen_data_loader.num_batch):
            inp_batch, image_batch = gen_data_loader.next_batch()
            feed_dict = {
                    mobilenet.X: image_batch,
                    mobilenet.is_training: False 
            }
            hidden_batch = sess.run(mobilenet.y_output, feed_dict=feed_dict)
            samples.extend(generator.generate(sess, hidden_batch).tolist())
        dis_data_loader.create_batches(random.sample(datas, 3000), samples)
        dis_data_loader.reset_pointer()
        for it in range(dis_data_loader.num_batch):
            x_batch, labels = dis_data_loader.next_batch()
            feed = {
                discriminator.input_x: x_batch,
                discriminator.labels: labels,
                discriminator.dropout_keep_prob: 0.75
            }
            _ = sess.run(discriminator.train_op, feed)

    # final test
    gen_data_loader.reset_pointer()
    _, image_batch = gen_data_loader.next_batch()
    feed_dict = {
                    mobilenet.X: image_batch,
                    mobilenet.is_training: False 
                }
    hidden_batch = sess.run(mobilenet.y_output, feed_dict=feed_dict)
    samples = generator.generate(sess, hidden_batch)
    y = samples.tolist()
    sams = []
    for k, sam in enumerate(y):
        sa = [id_to_word[i] for i in sam]
        sa = ''.join(sa)
        sams.append(sa)
    for sam in sams:
        log.write(sam+'\n')
    log.close()


if __name__ == '__main__':
    main()
