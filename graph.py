from keras.models import Sequential, Graph
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
import numpy as np

print "Loading Dataset.."
train_in = np.load('training_data/train_in.npy')
train_out = np.load('training_data/train_out.npy')
emotion_in = np.load('training_data/doc_in.npy')

assert len(train_in) == len(train_out)
assert len(train_out) == len(emotion_in)

print "Shuffling data..."
p = np.random.permutation(len(train_in))
train_in = train_in[p]
train_out = train_out[p]
emotion_in = emotion_in[p]

numwords = np.max(train_in)+1
graph = Graph()

graph.add_input(name='emotion_vector', input_shape=((1,)), dtype='int')
graph.add_node(Embedding(10000, 2, input_length=1), name='emotion_embeddings', input='emotion_vector')
graph.add_node(Flatten(), input='emotion_embeddings', name='flatemot')


graph.add_input(name='word_input', input_shape=((16,)), dtype='int')
graph.add_node(Embedding(numwords, 128, input_length=16), input='word_input', name='word_embeddings')
graph.add_node(Flatten(), input='word_embeddings', name='flatwords')

graph.add_node(Dense(512), inputs=['flatemot', 'flatwords'], name='dense1')
graph.add_node(Activation('relu'), input='dense1', name='activation2')

graph.add_node(Dense(256), input='activation2', name='a22')
graph.add_node(Activation('relu'), input='a22', name='a22a')

graph.add_node(Dense(numwords), input='a22a', name='out')
graph.add_node(Activation('softmax'), input='out', name='softmax')
graph.add_output(name='output', input='softmax')

a = Adam(lr=0.00001)
graph.compile(optimizer=a, loss={'output':'categorical_crossentropy'})

for e in range(0, 50):
    CE = 0.0
    i = 0
    j = 0
    batch_size = 512
    while i+batch_size < len(train_in):
        ein = emotion_in[i:i+batch_size].reshape(batch_size, 1)
        trin = train_in[i:i+batch_size]
        tout = np.zeros((batch_size, numwords))
        c = 0
        for t in train_out[i:i+batch_size]:
            tout[c][t] = 1
            c += 1
        CE += graph.train_on_batch({'emotion_vector':ein, 'word_input':trin, 'output':tout})[0]
        i += batch_size
        j += 1
        print 'Percent Done: ', float(i)/len(train_in), "CE: ", CE/j
    model.save_weights(str(e)+'a')
