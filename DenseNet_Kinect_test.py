'''

Testing phase,
input: 200*200 images contained one single person,
output: a 1*22 vector, which is the 11 keypoints coordinate

very simple,
very naive,
and very raw.

zhangboshen
2017.9.21
'''

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
import scipy.io as scio
import matplotlib.image as mpimg
import os
import time
time0 = time.time()

def conv_layer(input, filters,kernel_size,stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        net = slim.conv2d(input, filters, kernel_size, scope=layer_name)
        return net

class DenseNet():
    def __init__(self,x,nb_blocks, filters, sess):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.model = self.build_model(x)
        self.sess = sess
    
    def bottleneck_layer(self,x, scope):
        # [BN --> ReLU --> conv11 --> BN --> ReLU -->conv33]
        with tf.name_scope(scope):
            x = slim.batch_norm(x)
            x = tf.nn.relu(x)
            x = conv_layer(x,self.filters,kernel_size=(1,1), layer_name=scope+'_conv1')
            x = slim.batch_norm(x)
            x = tf.nn.relu(x)
            x = conv_layer(x,self.filters,kernel_size=(3,3), layer_name=scope+'_conv2')
            return x 
    def transition_layer(self,x, scope):
        # [BN --> conv11 --> avg_pool2]
        with tf.name_scope(scope):
            x = slim.batch_norm(x)
            x = conv_layer(x,self.filters,kernel_size=(1,1), layer_name=scope+'_conv1')
            x = slim.avg_pool2d(x,2)
            return x 
    
    def dense_block(self,input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = []
            layers_concat.append(input_x)
            x = self.bottleneck_layer(input_x,layer_name +'_bottleN_'+str(0))
            layers_concat.append(x)
            for i in xrange(nb_layers):
                x = tf.concat(layers_concat,axis=3)
                x = self.bottleneck_layer(x,layer_name+'_bottleN_'+str(i+1))
                layers_concat.append(x)
            return x
        
    
    def build_model(self,input_x):
        x = conv_layer(input_x,self.filters,kernel_size=(7,7), layer_name='conv0')
        x = slim.max_pool2d(x,(2,2))        
        for i in xrange(self.nb_blocks):
            x = self.dense_block(x,4, 'dense_'+str(i))
            x = self.transition_layer(x,'trans_'+str(i))       
        flatten = slim.flatten(x)
        x = tf.contrib.layers.fully_connected(flatten, 4096, scope='fc1')
        x = tf.contrib.layers.fully_connected(x, 22, activation_fn=None, scope='fc2')
        return x
		

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
batch_size = 1
image_number = 182   # hom many testing images you have
sess = tf.Session()

# load images
img_dir = 'where you put your training images, these images should have name: 1.png, 2.png,..........'
image = np.ones(image_number*200*200*3,dtype='double').reshape(image_number,200,200,3)
for i in range(0,image_number):
    image_name = str(i+1)+'.png'
    IMG_name = os.path.join(img_dir,image_name)
    img_temp = mpimg.imread(IMG_name)
    image[i] = img_temp
print ('image_type',np.shape(image))


def next_batch(batch,label):
    images = np.ones(200*200*3*batch,dtype='double').reshape(batch,200,200,3)  #NHWC
    idx=np.arange(0,len(label))
    np.random.shuffle(idx)
    idx=idx[0:batch]
    images = [image[i] for i in idx]
    images = np.asarray(images)
    label = [label[i] for i in idx]   
    label = np.asarray(label)

    return images,label,idx

def load_image(img_name):
    image = mpimg.imread(img_name)
    return image

images_source = tf.placeholder(tf.float32, shape=[batch_size, 200,200,3])
labels_source = tf.placeholder(tf.float32, shape=[batch_size, 22])   #label keypoints

labels_source = tf.cast(labels_source, tf.float32)

global_step = tf.contrib.framework.get_or_create_global_step()
model = DenseNet(images_source, 3, 12, sess)   # 3 blocks; 12 growth rate

pred = model.build_model(images_source)        # predictions
loss = tf.reduce_sum( tf.square(pred - labels_source))

trainable_variables = tf.trainable_variables()
grads = tf.gradients(loss, trainable_variables)

lrn_rate = 0.0000001

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

saver.restore(sess, "where you saved your madel")

Des_result = np.ones(image_number*22).reshape(image_number,22)
time1 = time.time()
for i in range(0, image_number):
    image_data = image[i].reshape((1,200,200,3))
    Des_result[i] = sess.run(pred,feed_dict={images_source:image_data})
time2 = time.time()
print 'done'
print ('initial_time',time1 - time0)
print ('total_time',time2 - time1)
# save data
scio.savemat('save your result as a .mat file',{'Des_result':Des_result})


 
