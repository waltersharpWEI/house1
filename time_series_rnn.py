
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

matplotlib.rcParams['figure.figsize'] = (16.0, 8.0)


# In[2]:


X = pd.read_csv('House_info_data.csv',encoding='gbk')
Y = pd.read_csv('House_price_data.csv',encoding='gbk')


# In[3]:


data = X.loc[:,['HOUSE_SRN','CITY','AREA']]
Y.HOUSE_SRN = Y.HOUSE_SRN.astype(str)
total = pd.merge(data,Y,on="HOUSE_SRN",how='left')
total['DATA_DATE'] = pd.to_datetime(total['DATA_DATE'])
total['DATA_DATE_Y'] = total.DATA_DATE.dt.year
total['DATA_DATE_M'] = total.DATA_DATE.dt.month
total['DATA_DATE_RM'] = (total.DATA_DATE - pd.to_datetime('2004/01/01')).astype('timedelta64[M]') + 1
total = total.drop('HIGHEST_PRICE',axis=1)
total = total.drop('LOWEST_PRICE',axis=1)
total.dropna(how='any')
total.head()


# In[4]:


samp = total[total['AREA'] == u'朝阳'].groupby(['DATA_DATE_RM']).mean().reset_index()
samp.head()


# In[77]:


samp.tail()


# In[5]:


# split into trainset and testset
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


length = samp.shape[0]
a = samp.DATA_DATE_RM.values.reshape(length, 1)
samples = samp['AVG_PRICE'].astype(np.float32).values
plt.plot(a,samples.reshape(length,1),'-o')
plt.show()
min_max_scaler = preprocessing.MinMaxScaler()
samples = min_max_scaler.fit_transform(samples)
plt.plot(a,samples.reshape(length,1),'-o')
plt.show()


# In[73]:


import tensorflow as tf

truncated_backprop_length = 15
input_size = 5
state_size = 4
num_classes = 1
# echo_step = 3
batch_size = 1
# num_batches = samples.shape[0]//batch_size
num_epochs = 10000
truncated_num = truncated_backprop_length - input_size + 1


# In[38]:


batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_num])

#and one for the RNN state, 5,4 
init_state = tf.placeholder(tf.float32, [batch_size, state_size])

#3 layer recurrent net, one hidden state

#randomly initialize weights
W = tf.Variable(np.random.rand(state_size+input_size, state_size), dtype=tf.float32)
#anchor, improves convergance, matrix of 0s 
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)
labels_series = tf.unstack(batchY_placeholder, axis=1)


# In[39]:


#Forward pass
#state placeholder
current_state = init_state
#series of states through time
states_series = []

#for each set of inputs
#forward pass through the network to get new state value
#store all states in memory
for i in range(truncated_num):
    current_input = batchX_placeholder[:,i:i + input_size]
#     mix input and state
    input_and_state_concatenated = tf.concat([current_input,current_state],1)
    #perform matrix multiplication between weights and input, add bias
    #squash with a nonlinearity, for probabiolity value
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    #store the state in memory
    states_series.append(next_state)
    #set current state to next one
    current_state = next_state
    


# In[68]:


#calculate loss
#second part of forward pass
#logits short for logistic transform
output_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
losses = [tf.squared_difference(output,labels) for output, labels in zip(output_series,labels_series)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(0.003).minimize(total_loss)


# In[87]:


#Step 3 Training the network
with tf.Session() as sess:
    #we stupidly have to do this everytime, it should just know
    #that we initialized these vars. v2 guys, v2..
    sess.run(tf.global_variables_initializer())
    #interactive mode
    plt.ion()
    #initialize the figure
    plt.figure()
    #show the graph
    plt.show()
    #to show the loss decrease
    loss_list = []
    
    x = samples.reshape(batch_size,samples.shape[0]//batch_size)
    
    for epoch_idx in range(num_epochs):
        #initialize an empty hidden state
        _current_state = np.zeros((batch_size, state_size))

#         print("New data, epoch", epoch_idx)
        #each batch
        for batch_idx in range(x.shape[1] // truncated_backprop_length):
            #starting and ending point per batch
            #since weights reoccuer at every layer through time
            #These layers will not be unrolled to the beginning of time, 
            #that would be too computationally expensive, and are therefore truncated 
            #at a limited number of time-steps
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length
            
            batchX = x[:,start_idx : end_idx]
            batchY = x[:,start_idx+input_size:start_idx+input_size+truncated_num]

            #run the computation graph, give it the values
            #we calculated earlier
            _total_loss, _train_step, _current_state, _output_series = sess.run(
                [total_loss, train_step, current_state, output_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state
                })
            
            loss_list.append(_total_loss)
        
        if epoch_idx%100 == 0:
            print("Step",epoch_idx, "Loss", _total_loss)
#             plt.plot(loss_list)
    plt.plot(loss_list)
    plt.show()
    
    #evaluate
    #initialize an empty hidden state
    _current_state = np.zeros((batch_size, state_size))
    in_list = []
    out_list = []
#         print("New data, epoch", epoch_idx)
        #each batch
    for batch_idx in range(x.shape[1] // truncated_backprop_length):
        start_idx = batch_idx * truncated_backprop_length
        end_idx = start_idx + truncated_backprop_length
            
        batchX = x[:,start_idx : end_idx]
        batchY = x[:,start_idx+input_size:start_idx+input_size+truncated_num]

        #run the computation graph, give it the values
        #we calculated earlier
        _total_loss, _train_step, _current_state, _output_series = sess.run(
            [total_loss, train_step, current_state, output_series],
            feed_dict={
                batchX_placeholder:batchX,
                batchY_placeholder:batchY,
                init_state:_current_state
            })
        in_list = np.concatenate([in_list,batchY[0]])
        out_list = np.concatenate([out_list,np.dstack(_output_series)[0][0]])
#         out_list = out_list + _output_series
        
    plt.plot(out_list)
    plt.plot(in_list)
    plt.show()
    
    #predicting
    batchX = x[:,end_idx:-1]
    print(batchX)
    batchX = np.array([[0.62848353,0.63562942,0.88827062 , 0.65884048 ,0 ,0, 0 ,0, 0 ,0, 0 ,0 ,0 ,0, 0]])
    batchY = np.array([[0,0,0,0,0,0,0,0,0,0,0]])
    print(batchX)
    _total_loss, _current_state, _output_series = sess.run(
        [total_loss, current_state, output_series],
        feed_dict={
            batchX_placeholder:batchX,
            batchY_placeholder:batchY,
            init_state:_current_state
        })
    print(_output_series)

plt.ioff()
plt.show()
            


# In[76]:


MSE = sum((in_list - out_list) ** 2)
MSE


# In[88]:


np.dstack(_output_series)[0][0]


# In[91]:


min_max_scaler.inverse_transform(np.array([0.52195925]))

