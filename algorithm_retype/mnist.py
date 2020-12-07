import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
tf.logging.set_verbosity(tf.logging.ERROR)
tf=tf.compat.v1
mnist = input_data.read_data_sets("../dataset/mnist/", one_hot=True)
#加载数据
def getData():
 print("Download and Extract MNIST dataSet")
 mnist = input_data.read_data_sets("../dataset/mnist/", one_hot=True)
 print("type of mnist is %s" % (type(mnist)))
 print("number of train data is %d" % (mnist.train.num_examples))
 print("number of test data is %d" % (mnist.test.num_examples))
 train_imgs = mnist.train.images
 train_labels = mnist.train.labels
 test_imgs = mnist.test.images
 test_labels = mnist.test.labels
 return train_imgs,train_labels,test_imgs, test_labels

if __name__=="__main__":
 print("Loading Data")
 train_imgs,train_labels,test_imgs, test_labels=getData()
 print("训练图片的形状：",train_imgs.shape," 训练标签的形状：",train_labels.shape," 测试图片的形状：",test_imgs.shape," 测试标签的形状：",test_labels.shape)
 print("load Data complete")

 #数据图像化
 n_samples=5
 #_random=np.random.randint(train_imgs.shape[0],size=n_samples)
 for j in range(n_samples):
  i=np.random.randint(train_imgs.shape[0])
  curr_img=np.reshape(train_imgs[i],(28,28))
  curr_labs=np.argmax(train_labels[i])#令人纳闷：这里取出来是应该是一个数组，可为什么变成3了。我知道了前面有argmax()->取最大值的索引
  plt.matshow(curr_img,cmap=plt.get_cmap('gray'))#灰度图
  plt.title(str(i)+'th Training Data,Label is '+str(curr_labs))#这里不加str会出现numpy64不能被迭代错误。把加号换成逗号也会报错
  plt.show()

 #batch_size学习
 batch_size=100
 batch_xs,batch_ys=mnist.train.next_batch(batch_size)
 print(type(batch_xs))
 print(batch_xs.shape)
 print(type(batch_ys))
 print(batch_ys.shape)
