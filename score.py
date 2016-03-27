from keras.models import Sequential, Graph
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
import numpy as np
import cPickle
from sklearn.neighbors import NearestNeighbors

train_in = np.load('training_data/train_in.npy')
train_out = np.load('training_data/train_out.npy')
wind = cPickle.load(open('training_data/wind', 'r'))

pos_in, pos_out, neg_in, neg_out = cPickle.load(open('pg', 'r'))

numwords = np.max(train_in[:, range(1, len(train_in[0]))])+1
graph = Graph()

graph.add_input(name='emotion_vector', input_shape=((1,)), dtype='int')
graph.add_node(Embedding(4096*2, 2, input_length=1), name='emotion_embeddings', input='emotion_vector')
graph.add_node(Flatten(), input='emotion_embeddings', name='flatemot')


graph.add_input(name='word_input', input_shape=((7,)), dtype='int')
graph.add_node(Embedding(numwords, 512, input_length=7), input='word_input', name='word_embeddings')
graph.add_node(Flatten(), input='word_embeddings', name='flatwords')

graph.add_node(Dense(1024), inputs=['flatemot', 'flatwords'], name='dense1')
graph.add_node(Activation('relu'), input='dense1', name='activation1')

graph.add_node(Dense(512), input='activation1', name='dense2')
graph.add_node(Activation('relu'), input='dense2', name='activation2')

graph.add_node(Dense(numwords), input='activation2', name='out')
graph.add_node(Activation('softmax'), input='out', name='softmax')
graph.add_output(name='output', input='softmax')

a = Adam(lr=0.0001)
graph.compile(optimizer=a, loss={'output':'categorical_crossentropy'})

graph.load_weights('49b')


tin = np.array(pos_in[0])
tout = np.zeros((len(pos_in[0]), numwords))





def classify_pos(tin, tout):
    test_in = np.array(tin)
    test_out = np.zeros((len(tin), numwords))
    for t in range(len(test_out)):
        test_out[t][tout[t][0]] = 1
    posloss = graph.evaluate(test_in, test_out, batch_size=512, verbose=0)
    test_in[:, 0] = wind['NEG_TEXT']
    negloss = graph.evaluate(test_in, test_out, batch_size=512, verbose=0)
    return posloss, negloss

FP = 0
TN = 0
i = 0
for pin, pout in zip(neg_in, neg_out):
    test_in = np.array(pin)
    if len(test_in) > 0:
        win = test_in[:, range(1, len(test_in[0]))]
        ein = test_in[:, 0].reshape(len(test_in), 1)
        ein[:,0] = 0
        test_out = np.zeros((len(pin), numwords))
        for t in range(len(test_out)):
            test_out[t][pout[t][0]] = 1
        posloss = graph.evaluate({'emotion_vector': ein, 'word_input': win, 'output': test_out}, batch_size=512)
        ein[:,0] = 1
        negloss = graph.evaluate({'emotion_vector': ein, 'word_input': win, 'output': test_out}, batch_size=512)
        if posloss > negloss:
            TN += 1
        else:
            FP += 1
        i += 1
        if i % 250 == 0:
            print i, TN, FP

FN = 0
TP = 0
i = 0
for pin, pout in zip(pos_in, pos_out):
    test_in = np.array(pin)
    if len(test_in) > 0:
        win = test_in[:, range(1, len(test_in[0]))]
        ein = test_in[:, 0].reshape(len(test_in), 1)
        ein[:,0] = 0
        test_out = np.zeros((len(pin), numwords))
        for t in range(len(test_out)):
            test_out[t][pout[t][0]] = 1
        posloss = graph.evaluate({'emotion_vector': ein, 'word_input': win, 'output': test_out}, batch_size=512)
        ein[:,0] = 1
        negloss = graph.evaluate({'emotion_vector': ein, 'word_input': win, 'output': test_out}, batch_size=512)
        if negloss > posloss:
            TP += 1
        else:
            FN += 1
        i += 1
        if i % 250 == 0:
            print i, float(TP+TN)/float(TP+TN+FP+FN), TP, FN


nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(graph.get_weights()[0])
rev_cab = {wind[x]: x for x in wind.keys()}

def get_n(s):
    index = wind[s]
    distances, indices = nbrs.kneighbors(graph.get_weights()[0][index])
    for x in indices[0]:
        print rev_cab[x]

get_n('three')
