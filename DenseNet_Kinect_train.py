
'''
This code,
1, Define a DenseNet class;
2, Use is to do a human pose estimation task(11 keypoints)).

input: a depth image collected by Kinect V2
(3 channels, normalized to 0~255, cropped to 200*200, contained one single person)

This is a keypoints regression task, 
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
        x = slim.flatten(x,scope='flatten1')
        x = slim.fully_connected(x, 4096, scope='fc1')
        x = slim.dropout(x, 0.5, scope='dropout1')
        x = slim.fully_connected(x, 22, scope='fc2')     # 11 keypoints
        return x
		

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
batch_size = 8
image_number = 159285   # hom many training images you have
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

# Load label
print('loading label')
labelfile = 'your skeleton_label file dir/skeleton_label.mat, which is a .mat file, image_number*keypoints_num*2'
label_mat = scio.loadmat(labelfile)
label = label_mat['skeleton_label']
label_data = label.reshape((len(label),22))
print('label_shape',np.shape(label_data))

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
loss = tf.reduce_sum( tf.square(pred - labels_source))   # L2 norm loss

trainable_variables = tf.trainable_variables()
grads = tf.gradients(loss, trainable_variables)

lrn_rate = 0.0000001

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
  train_op = tf.train.GradientDescentOptimizer(lrn_rate).minimize(loss)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

saver.restore(sess, 'where u saved u checkpoints files, for re-training phase')

#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # To shut down poolallocator information
for i in range(1000001):
 train_image ,train_label,idx = next_batch(batch_size,label_data)
 
 if i%100 == 0:
   distance = loss.eval(session=sess,feed_dict={images_source:train_image,labels_source:train_label})
   distance = tf.reduce_mean(tf.cast(distance, tf.float32))
   print('step:',i,'loss', sess.run(distance/batch_size),'image_index',idx)
 elif i% 1001 == 0:
   print "step %d,loss"%i
   print('predictions',sess.run(pred,feed_dict={images_source:train_image,labels_source:train_label}))
   print('ground_truth',train_label)
 # training prosess
 train_op.run(session=sess,feed_dict={images_source:train_image,labels_source:train_label})
 if i == 1000000:
   saver.save(sess,"where your want save your model")
 


