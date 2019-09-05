import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
x_data = np.linspace(0.0,10.0,1000000)
noise = np.random.randn(len(x_data))

# y = mx + b + noise_levels
b = 5

y_true =  (0.5 * x_data ) + 5 + noise

#my_data = pd.concat([pd.DataFrame(data=x_data,columns=['X Data']),pd.DataFrame(data=y_true,columns=['Y'])],axis=1)

import tensorflow as tf

# Random 10 points to grab
batch_size = 8

m = tf.Variable(0.5)
b = tf.Variable(1.0)

xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])

y_model = xph*m + b

error = tf.reduce_sum(tf.square(yph-y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)
    
    batches = 1000
    
    for i in range(batches):
        
        rand_ind = np.random.randint( len(x_data), size=batch_size ) # shape of rand_ind: (8,)
        
        feed = {
            xph:x_data[rand_ind],
            yph:y_true[rand_ind]
        }
        
        sess.run(train,feed_dict=feed)
        
    model_m,model_b = sess.run([m,b])

    print('model_m= '+str(model_m)+' model_b='+str(model_b) )