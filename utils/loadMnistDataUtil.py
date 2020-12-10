import tensorflow.examples.tutorials.mnist.input_data as input_data#input会调用一个maybe_download的成功，确保数据下载成功，如果已下载，就不再下载
def getMnistData():
    print("-" * 30, '数据加载', "-" * 30, '\n数据形状:')
    mnist = input_data.read_data_sets("../dataset/mnist", one_hot=True)
    train_imgs = mnist.train.images#调用时这里不能加括号
    train_labs = mnist.train.labels
    test_imgs = mnist.test.images
    test_labs = mnist.test.labels
    print("训练图片:", train_imgs.shape, "训练标签:", train_labs.shape)
    print("测试图片:", test_imgs.shape, "测试标签", test_labs.shape, '\n', "-" * 30, '数据加载完成', "-" * 30)

    return train_imgs,train_labs,test_imgs,test_labs,mnist
