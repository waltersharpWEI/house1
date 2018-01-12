
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
# import pyflux as pf
import tensorflow as tf
import os



# In[3]:


total =  pd.read_csv('avg_price.txt',encoding='gbk')
samp = total[total['AREA'] == u'朝阳'].groupby(['DATA_DATE_RM']).mean().reset_index()

samp.head()


# In[8]:


from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

length = samp.shape[0]
index = samp.DATA_DATE_RM.values
samples = samp['AVG_PRICE'].astype(np.float32).values.reshape(length,1)
#print(samples)
print('Data processed...')
min_max_scaler = preprocessing.MinMaxScaler()
samples = min_max_scaler.fit_transform(samples)


# <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png">
# LSTM
# $$hf_{t}=\sigma(W_{f}[h_{t-1},x_{t}] + b_{f})$$
# 
# $$hi_{t}=\sigma(W_{i}[h_{t-1},x_{t}] + b_{i})$$
# 
# $$ho_{t}=\sigma(W_{o}[h_{t-1},x_{t}] + b_{o})$$
# 
# $$hc_{t}=\tanh(W_{c}[h_{t-1},x_{t}] + b_{c})$$
# 
# $$c_{t}=hf_{t}\times c_{t-1} + hi_{t}\times hc_{t}$$
# 
# $$h_{t}=ho_{t}\times \tanh (c_{t})$$

# In[17]:


H = 256 # Number of LSTM layer's neurons
D = 1 # Number of input dimension == number of items in vocabulary
Z = H + D # Because we will concatenate LSTM state with the input
time_step = 30 #Number of time_step to backprop
num_epochs = 10000
learning_rate = 0.001

tf.reset_default_graph()

Wf = tf.Variable(np.random.randn(Z, H) / np.sqrt(Z / 2.),dtype=tf.float32,name='Wf')
Wi = tf.Variable(np.random.randn(Z, H) / np.sqrt(Z / 2.),dtype=tf.float32,name='Wi')
Wc = tf.Variable(np.random.randn(Z, H) / np.sqrt(Z / 2.),dtype=tf.float32,name='Wc')
Wo = tf.Variable(np.random.randn(Z, H) / np.sqrt(Z / 2.),dtype=tf.float32,name='Wo')
Wy = tf.Variable(np.random.randn(H, D) / np.sqrt(D / 2.),dtype=tf.float32,name='Wy')

bf = tf.Variable(np.zeros((1,H)),dtype=tf.float32,name='bf')
bi = tf.Variable(np.zeros((1,H)),dtype=tf.float32,name='bi')
bc = tf.Variable(np.zeros((1,H)),dtype=tf.float32,name='bc')
bo = tf.Variable(np.zeros((1,H)),dtype=tf.float32,name='bo')
by = tf.Variable(np.zeros((1,D)),dtype=tf.float32,name='by')


#init h value
init_h = tf.placeholder(tf.float32, [1,H],name='init_h')
init_c = tf.placeholder(tf.float32, [1,H],name='init_c')
X_placeholder = tf.placeholder(tf.float32,[1,time_step],name='X_placeholder')
Y_placeholder = tf.placeholder(tf.float32,[1,time_step],name='Y_placeholder')
labels_series = tf.unstack(Y_placeholder, axis=1)


# In[18]:


current_h = init_h
current_c = init_c
Y_predict_series = []
# lstm forword
for i in range(time_step):
    #1*D matrix
    current_input = X_placeholder[:,i:i + D]
    # mix input and hidden units; 1*Z matrix
    input_and_h_concatenated = tf.concat([current_input,current_h],1) # 1*Z matrinx
    hf = tf.sigmoid(tf.matmul(input_and_h_concatenated,Wf) + bf) # 1*H matrix
    hi = tf.sigmoid(tf.matmul(input_and_h_concatenated,Wi) + bi) # 1*H matrix
    ho = tf.sigmoid(tf.matmul(input_and_h_concatenated,Wo) + bo) # 1*H matrix
    hc = tf.tanh(tf.matmul(input_and_h_concatenated,Wc) + bc) #1*H matrix
    
    #set current state
    current_c = hf * current_c + hi * hc 
    current_h = ho * tf.tanh(current_c) #1*H matrix
    Y_predict = tf.sigmoid(tf.matmul(current_h,Wy) + by)
    # store Y prediction in memery
    Y_predict_series.append(Y_predict)
    


# In[19]:


#calculate loss
losses = [tf.squared_difference(output,Y) for output,Y in zip(Y_predict_series,labels_series)]
total_loss = tf.reduce_mean(losses)
tf.summary.scalar("loss", total_loss)
train_step = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(total_loss)


# In[ ]:


def make_hparam_string(learning_rate, HiddenUnits):
    return "lr_%.0E_%d" % (learning_rate, HiddenUnits)


# In[ ]:


# training model
LOGDIR = 'tensorboardlog/'
with tf.Session() as sess:

    #interactive mode
    #initialize the figure
    #show the graph

    hparam = make_hparam_string(learning_rate,H)
    summ = tf.summary.merge_all()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    #filewriter is how we write the summary protocol buffers to disk
    writer = tf.summary.FileWriter(LOGDIR + hparam)
    writer.add_graph(sess.graph)

    #to show the loss decrease
    loss_list = []

    # orginal 155 points,discard first 4 element, make it 151 points to predict
    x = samples[4:].reshape(1,151)
#     for i in range(x.shape[1]):
#         x[0,i] = (x[0,i]/x[0,0])-1
#     print(x)
    for epoch_idx in range(num_epochs):
        #initialize an empty hidden state
        _current_c = np.zeros((1, H))
        _current_h = np.zeros((1, H))
        y_predict_list = []
    #         print("New data, epoch", epoch_idx)
        #each batch
        for batch_idx in range(x.shape[1] // time_step):
            #starting and ending point per batch
            #since weights reoccuer at every layer through time
            #These layers will not be unrolled to the beginning of time, 
            #that would be too computationally expensive, and are therefore truncated 
            #at a limited number of time-steps
            start_idx = batch_idx * time_step
            end_idx = start_idx + time_step 
            x_data = x[:,start_idx : end_idx]
            y_data = x[:,start_idx + D:end_idx + D]

            #run the computation graph, give it the values
            #we calculated earlier
            _total_loss, _train_step, _current_c, _current_h, _Y_predict_series,s = sess.run(
                [total_loss, train_step, current_c,current_h, Y_predict_series,summ],
                feed_dict={
                    X_placeholder:x_data,
                    Y_placeholder:y_data,
                    init_h:_current_h,
                    init_c:_current_c
                })

            loss_list.append(_total_loss)
            y_predict_list.append(_Y_predict_series)

        if epoch_idx%5==0:
            writer.add_summary(s, epoch_idx)
        if epoch_idx%100 == 0:
            print("Step",epoch_idx, "Loss", _total_loss)
            #save checkpoints
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)


# In[14]:


y_output = []
np.dstack(y_predict_list[0])[0][0]
for series in y_predict_list:
    y_output = np.concatenate([y_output,np.dstack(series)[0][0]])
y_output.shape


# In[15]:




# In[16]:


y_output.shape

