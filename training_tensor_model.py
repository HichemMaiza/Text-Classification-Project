import pandas as pd 
import numpy as np 
import tensorflow as tf
from tensorflow.python.framework import ops

data_train = pd.read_pickle('data_train')
data_dev = pd.read_pickle('data_dev')
data_test = pd.read_pickle('data_test')
data_test_auto = pd.read_pickle('data_test_auto')

wordsList = np.load('wordsList.npy')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordVectors = np.load('wordVectors.npy')
numDimensions=350
maxSeqLength = 350
ids = np.load('idsMatrix_train_dev.npy')
ids_test = np.load('idsMatrix_test.npy')
ids_test_auto = np.load('idsMatrix_test_auto.npy')


print(' ids are loaded') 


from sklearn import preprocessing
le = preprocessing.LabelEncoder() 
le.fit(data_train['file_label'].append(data_dev['file_label'])) 
numClasses = len(le.classes_)
int_labels = le.transform(data_train['file_label'].append(data_dev['file_label']))

int_labels_test = le.transform(data_test['file_label'])

int_labels_test_auto = le.transform(data_test_auto['file_label'])
#int_labels = int_labels.reshape(1,-1)

def convert_to_onehot(y, nbclass):  
    '''
    Creates a Matrix of one hot vector from (Y)
    
    Arguments:
    Y -- input classes, of shape (1, number of examples)
    nbclass -- number of Classes
    
    Returns:
    y_train -- Matrix coded in one hot logic
    '''
    y_train = [] 
    for i in range(y.shape[0]) :
        temp =np.zeros((nbclass))
        temp[y[i]] = 1
        y_train.append(temp)
    return np.array(y_train)
labels = convert_to_onehot(int_labels, 14)
labels_test = convert_to_onehot(int_labels_test, 14)
labels_test_auto = convert_to_onehot(int_labels_test_auto, 14)


print('the shape of labels is ', labels.shape) 
batchSize = 64
lstmUnits = 64
#tches(X, Y, mini_batch_size = 64, seed = 0)iterations = 2

# compute random batches
def random_batches(X, Y, mini_batch_size = 64, seed = 0) :
    import math
    import numpy as np
    #Y = Y.reshape(-1,1)
    np.random.seed(seed)            # to get the same result every time so i can verify my results       
    m = X.shape[0]                  # number of training examples
    mini_batches = []      
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m)) ; #print(permutation)
    shuffled_X = X[permutation,:] ; print(X.shape) ; print(Y.shape)
    shuffled_Y = Y[permutation,:]# .reshape((m, Y.shape[1]))
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.d
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k*mini_batch_size:k*mini_batch_size + mini_batch_size,: ]
        mini_batch_Y = shuffled_Y[k*mini_batch_size:k*mini_batch_size + mini_batch_size, : ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches*mini_batch_size:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches*mini_batch_size:,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)    
    return mini_batches

a = random_batches(ids, labels, mini_batch_size = 64, seed = 0) 
t = random_batches(ids_test, labels_test, mini_batch_size = 64, seed = 0)

# define placeholder
labels = tf.placeholder(tf.float32, [None, numClasses])
input_data = tf.placeholder(tf.int32, [None, maxSeqLength])
# define embeddings 
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)
# define model architecture
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
# calculate accuracy using correct prediction
prediction = (tf.matmul(last, weight) + bias)
correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
# define cost function (a cross entroy loss function)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)
# begin tensorflow session
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
seed = 0
# tain the model
for epoch in range(3):
    epoch_cost = 0. ; num_minibatches = int(len(int_labels) / 64)
    #num_minibatches = int(len(int_labels) / minibatch_size)       
    for minibatch in a:
        # Select a minibatch
        (minibatch_X, minibatch_Y) = minibatch
        _ , minibatch_cost = sess.run([optimizer, loss], feed_dict={input_data: minibatch_X, labels: minibatch_Y})
        epoch_cost += minibatch_cost / num_minibatches

    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
    if (epoch % 100 == 0 and epoch != 0):
        save_path = saver.save(sess, "pretrained_lstm.ckpt", global_step=epoch)
        print("saved to %s" % save_path)
# evalute the model
# calculate precison and recall of the system 
from sklearn.metrics import precision_score, recall_score
y_pred = tf.argmax(prediction, 1) 
val_accuracy, y_pred = sess.run([accuracy, y_pred], {input_data:ids_test, labels: labels_test})
y_true = np.argmax(labels_test, 1)
print("Accuracy manu: ", val_accuracy)    
macroPrecision = precision_score(y_true, y_pred, average='macro')
macroRecall = recall_score(y_true,y_pred, average='macro')
microPrecision = precision_score(y_true, y_pred, average='micro')
microRecall = recall_score(y_true, y_pred, average='micro')  
print('the microPrecision is ', microPrecision) 
print('the macroPrecision is ', macroPrecision)

print('the microRecall is ', microRecall)
print('the macroRecall is ', macroRecall)

