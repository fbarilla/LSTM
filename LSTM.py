import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import sys
sys.path.insert(0, "/opt/DL/tensorflow/lib/python2.7/site-packages/")
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

import matplotlib.pyplot as plt
# from tsc_model import Model,sample_batch,load_data,check_test

def load_data(direc,ratio,dataset):
  """Input:
  direc: location of the UCR archive
  ratio: ratio to split training and testset
  dataset: name of the dataset in the UCR archive"""
  datadir = direc + '/' + dataset
  data_train = np.loadtxt(datadir+'_TRAIN',delimiter=',')
  data_test_val = np.loadtxt(datadir+'_TEST',delimiter=',')
    
  DATA = np.concatenate((data_train,data_test_val),axis=0)
  N = DATA.shape[0]

  ratio = (ratio*N).astype(np.int32)
  ind = np.random.permutation(N)
  X_train = DATA[ind[:ratio[0]],1:]
  X_val = DATA[ind[ratio[0]:ratio[1]],1:]
  X_test = DATA[ind[ratio[1]:],1:]
  # Targets have labels 1-indexed. We subtract one for 0-indexed
  y_train = DATA[ind[:ratio[0]],0]-1
  y_val = DATA[ind[ratio[0]:ratio[1]],0]-1
  y_test = DATA[ind[ratio[1]:],0]-1
  return X_train,X_val,X_test,y_train,y_val,y_test


def sample_batch(X_train,y_train,batch_size):
  """ Function to sample a batch for training"""
  N,data_len = X_train.shape
  ind_N = np.random.choice(N,batch_size,replace=False)
  X_batch = X_train[ind_N]
  y_batch = y_train[ind_N]
  return X_batch,y_batch
  
  
def check_test(model,sess,X_test,y_test):
  batch_size = model.batch_size
  N = X_test.shape[0]
  runs = int(np.floor(float(N)/batch_size))
  
  perfs = []
  for i in xrange(runs):
    X_batch,y_batch = sample_batch(X_test,y_test,batch_size)
    result = sess.run([model.cost,model.accuracy],feed_dict = {model.input: X_batch, model.labels: y_batch, model.keep_prob:1.0})
    perfs.append(tuple(result))
  acc,cost = [np.mean(x) for x in zip(*perfs)]
  return acc,cost

class Model():
  def __init__(self,config):
    
    num_layers = config['num_layers']
    hidden_size = config['hidden_size']
    max_grad_norm = config['max_grad_norm']
    self.batch_size = config['batch_size']
    sl = config['sl']
    learning_rate = config['learning_rate']
    num_classes = config['num_classes']
    """Place holders"""
    self.input = tf.placeholder(tf.float32, [None, sl], name = 'input')
    self.labels = tf.placeholder(tf.int64, [None], name='labels')
    self.keep_prob = tf.placeholder("float", name = 'Drop_out_keep_prob')

    with tf.name_scope("LSTM_setup") as scope:
#      cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
#      cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
#      cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
      cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(hidden_size)
      cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
      cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell] * num_layers)
      initial_state = cell.zero_state(self.batch_size, tf.float32)
    
    input_list = tf.unstack(tf.expand_dims(self.input,axis=2),axis=1)
    # outputs,_ = tf.nn.seq2seq.rnn_decoder(input_list,initial_state,cell)
    outputs,_ = tf.contrib.legacy_seq2seq.rnn_decoder(input_list,initial_state,cell)

    output = outputs[-1]

 
    #Generate a classification from the last cell_output
    #Note, this is where timeseries classification differs from sequence to sequence
    #modelling. We only output to Softmax at last time step
    with tf.name_scope("Softmax") as scope:
      with tf.variable_scope("Softmax_params"):
        softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
        softmax_b = tf.get_variable("softmax_b", [num_classes])
      logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
      #Use sparse Softmax because we have mutually exclusive classes
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,self.labels,name = 'softmax')
      self.cost = tf.reduce_sum(loss) / self.batch_size
    with tf.name_scope("Evaluating_accuracy") as scope:
      correct_prediction = tf.equal(tf.argmax(logits,1),self.labels)
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    """Optimizer"""
    with tf.name_scope("Optimizer") as scope:
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),max_grad_norm)   #We clip the gradients to prevent explosion
      optimizer = tf.train.AdamOptimizer(learning_rate)
      gradients = zip(grads, tvars)
      self.train_op = optimizer.apply_gradients(gradients)
      # Add histograms for variables, gradients and gradient norms.
      # The for-loop loops over all entries of the gradient and plots
      # a histogram. We cut of
      for gradient, variable in gradients:  #plot the gradient of each trainable variable
            if isinstance(gradient, ops.IndexedSlices):
              grad_values = gradient.values
            else:
              grad_values = gradient

            tf.summary.histogram(variable.name, variable)
            tf.summary.histogram(variable.name + "/gradients", grad_values)
            tf.summary.histogram(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))

    #Final code for the TensorBoard
    self.merged = tf.summary.merge_all()
    self.init_op = tf.global_variables_initializer()

"""Hyperparamaters"""

config = {}                             #Put all configuration information into the dict
config['num_layers'] = 3                #number of layers of stacked RNN's
config['hidden_size'] = 120              #memory cells in a layer
config['max_grad_norm'] = 5             #maximum gradient norm during training
config['batch_size'] = batch_size = 30  
config['learning_rate'] = .005

max_iterations = 3000
dropout = 0.8
ratio = np.array([0.8,0.9]) #Ratios where to split the training and validation set

"""Load the data"""
direc = './'
X_train,X_val,X_test,y_train,y_val,y_test = load_data(direc,ratio,dataset='Two_Patterns')
N,sl = X_train.shape
config['sl'] = sl = X_train.shape[1]
config['num_classes'] = num_classes = len(np.unique(y_train))

# Collect the costs in a numpy fashion
epochs = np.floor(batch_size*max_iterations / N)
print('Train %.0f samples in approximately %d epochs' %(N,epochs))
perf_collect = np.zeros((4,int(np.floor(max_iterations /100))))

#Instantiate a model
model = Model(config)



"""Session time"""
sess = tf.Session() #Depending on your use, do not forget to close the session
writer = tf.summary.FileWriter("/root/frb/MLDL/Clean-Energy/log_tb", sess.graph)  #writer for Tensorboard
sess.run(model.init_op)

step = 0
cost_train_ma = -np.log(1/float(num_classes)+1e-9)  #Moving average training cost
acc_train_ma = 0.0
try:
  for i in range(max_iterations):
    X_batch, y_batch = sample_batch(X_train,y_train,batch_size)

    #Next line does the actual training
    cost_train, acc_train,_ = sess.run([model.cost,model.accuracy, model.train_op],feed_dict = {model.input: X_batch,model.labels: y_batch,model.keep_prob:dropout})
    cost_train_ma = cost_train_ma*0.99 + cost_train*0.01
    acc_train_ma = acc_train_ma*0.99 + acc_train*0.01
    if i%100 == 0:
      #Evaluate training performance
      perf_collect[0,step] = cost_train
      perf_collect[1,step] = acc_train

      #Evaluate validation performance
      X_batch, y_batch = sample_batch(X_val,y_val,batch_size)
      cost_val, summ,acc_val = sess.run([model.cost,model.merged,model.accuracy],feed_dict = {model.input: X_batch, model.labels: y_batch, model.keep_prob:1.0})
      perf_collect[1,step] = cost_val
      perf_collect[2,step] = acc_val
      print('At %5.0f/%5.0f: COST %5.3f/%5.3f(%5.3f) -- Acc %5.3f/%5.3f(%5.3f)' %(i,max_iterations,cost_train,cost_val,cost_train_ma,acc_train,acc_val,acc_train_ma))

      #Write information to TensorBoard
      writer.add_summary(summ, i)
      writer.flush()

      step +=1
except KeyboardInterrupt:
  #Pressing ctrl-c will end training. This try-except ensures we still plot the performance
  pass
  
acc_test,cost_test = check_test(model,sess,X_test,y_test)
epoch = float(i)*batch_size/N
print('After training %.1f epochs, test accuracy is %5.3f and test cost is %5.3f'%(epoch,acc_test,cost_test))

"""Additional plots"""
plt.plot(perf_collect[0],label='Train')
plt.plot(perf_collect[1],label = 'Valid')
plt.plot(perf_collect[2],label = 'Valid accuracy')
plt.axis([0, step, 0, np.max(perf_collect)])
plt.legend()
plt.show()

