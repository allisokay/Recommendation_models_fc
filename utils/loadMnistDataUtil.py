import tensorflow.examples.tutorials.mnist.input_data as input_data
def getMnistData():
    mnist = input_data.read_data_sets("../dataset/mnist", one_hot=True)
    train_imgs = mnist.train.images#调用时这里不能加括号
    train_labs = mnist.train.labels
    test_imgs = mnist.test.images
    test_labs = mnist.test.labels
    return train_imgs,train_labs,test_imgs,test_labs,mnist
