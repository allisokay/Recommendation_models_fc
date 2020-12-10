#注：这个是跟随视频写的代码，视频中准确率会提升，但我得到视频中那个讲述人的代码运行后准确率就没有了，不知道是我的原因还是视频，让人挺灰心的
import numpy as np
import tensorflow as tf
import utils.loadMnistDataUtil as mnistData
tf=tf.compat.v1
tf.logging.set_verbosity(tf.logging.ERROR)

'''加载mnist'''
train_imgs,train_labs,test_imgs,test_labs,mnist=mnistData.getMnistData()
'''设置隐层'''
#设定随机数量的值 28*28
n_input=784
#输出数量
n_output=10
#权重
wights={
    'wc1':tf.Variable(tf.random_normal([3,3,1,64],stddev=0.1)),#平方差间距为0.1
    'wc2':tf.Variable(tf.random_normal([3,3,64,128],stddev=0.1)),
    'wd1':tf.Variable(tf.random_normal([7*7*128,1024],stddev=0.1)),
    'wd2':tf.Variable(tf.random_normal([1024,n_output],stddev=0.1))
}
#偏置
bias={
    "bc1":tf.Variable(tf.random_normal([64],stddev=0.1)),
    "bc2":tf.Variable(tf.random_normal([128],stddev=0.1)),
    "bd1":tf.Variable(tf.random_normal([1024],stddev=0.1)),
    "bd2":tf.Variable(tf.random_normal([n_output],stddev=0.1))
}
#CNN Ready
def conv_basic(_input,_w,_b,_keepRatio):#_keepRation:保持数据信息比拟
    print("-"*30,"CNN开始构建","-"*30)
    #输入
    _input_r=tf.reshape(_input,shape=[-1,28,28,1])

    #卷积的第一个隐层
    #nnc.conv2d使用cnn的模式
    #卷积层1
    _conv1=tf.nn.conv2d(_input_r,_w.get("wc1"),strides=[1,1,1,1],padding="SAME")
    print('第一次卷积后的图片形状：',_conv1.shape)
    #_mean,_var=nn.moment(_conv1,[0,1,2])
    #激活层1
    _conv_relu1=tf.nn.relu(tf.nn.bias_add(_conv1,_b.get('bc1')))
    print('第一次卷积再激活后的图片形状：', _conv_relu1.shape)
    #池化层:最大池化
    _conv_pool1=tf.nn.max_pool(_conv_relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    print('第一次卷积再激活再最大池化后的图片形状：', _conv_pool1.shape)
    #使用dropout防止过拟合化,保持比例
    _pool_dr1=tf.nn.dropout(_conv_pool1,_keepRatio)
    print('第一次卷积再激活再最大池化再剪枝后的图片形状：', _pool_dr1.shape)
    #卷积层2
    _conv2=tf.nn.conv2d(_pool_dr1,_w.get("wc2"),strides=[1,1,1,1],padding='SAME')
    print('第二次卷积后的图片形状：', _conv2.shape)
    #激活层2
    _conv_relu2=tf.nn.relu(tf.nn.bias_add(_conv2,_b.get("bc2")))
    print('第二次卷积再激活后的图片形状：', _conv_relu2.shape)
    #池化层：也是最大池化
    _conv_pool2=tf.nn.max_pool(_conv_relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    print('第二次卷积再激活再最大池化后的图片形状：', _conv_pool2.shape)
    #仍然再使用一个dropout防止过拟合
    _pool_dr2=tf.nn.dropout(_conv_pool2,_keepRatio)
    print('第二次卷积再激活再最大池化再剪枝后的图片形状：', _pool_dr2.shape)

    #矢量化
    _dense1=tf.reshape(_pool_dr2,[-1,_w.get('wd1').get_shape().as_list()[0]])
    print('矢量化后的图片形状：', _dense1.shape)
    #全连接层1 矩阵相乘
    _fc1=tf.nn.relu(tf.add(tf.matmul(_dense1,_w.get('wd1')),_b.get('bd1')))
    print('全连接后的图片形状：', _fc1.shape)
    #仍然防止过拟合
    _fc_dr1=tf.nn.dropout(_fc1,_keepRatio)
    print('全连接再剪枝后的图片形状：', _fc_dr1.shape)
    # #全连接层2，也是矩阵相乘
    # _fc2=tf.nn.relu(tf.add(tf.matmul(_fc_dr1,_w["wd2"]),_b['bd2']))
    # #继续防止过拟合
    # _fc_dr2=tf.nn.dropout(_fc2,_keepRatio)

    #输出层
    out=tf.add(tf.matmul(_fc_dr1,_w.get('wd2')),_b.get('bd2'))
    print('输出的图片形状：', out.shape)
    print("-"*30,"CNN构建完成","-"*30)
    return  out

#图形准备
x=tf.placeholder(tf.float32,[None,n_input])#声明float节点
y=tf.placeholder(tf.float32,[None,n_output])
keepRatio=tf.placeholder(tf.float32)
_pred=conv_basic(x,wights,bias,keepRatio)
'''cost和op是为了更新权重矩阵weights和偏置'''
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred,labels=y))#softmax:单调增函数
#Adam优化算法，一个寻求全局最优点的优化算法，引入二次梯度校正
op=tf.train.AdamOptimizer(0.001).minimize(cost)
#op=tf.train.GradientDescentOptimizer(0.001).minimize(cost)
'''corr和accr是为了在权重weights和偏置bias更新完成后进行准确性的比较'''
#对比这两个矩阵或者向量相等的元素，如果相等返回True,否则返回False,返回的矩阵维度相同
_corr=tf.equal(tf.argmax(_pred,1),tf.argmax(y,1))#将预测值_pred和真实值y中的每一个值都做比较，相等返回True，否则False
#准确率
accr=tf.reduce_mean(tf.cast(_corr,tf.float32))#将_corr中的True转为1，False转为0，计算矩阵中所有值之和的均值得到模型的准确率
print("-"*30,'图形构建完成',"-"*30)

#初始化
init=tf.global_variables_initializer()
session=tf.Session()
session.run(init)

#开始训练
epochs=15
batch_size=16
display_step=1
for epoch in range(epochs):
    avg_cost=0.
    total_batch=10 #mnist.train.num_examples/batch_size
    for i in range(total_batch):
        batchXs,batchYs=mnist.train.next_batch(batch_size)
       # feeds={x:batchXs,y:batchYs,keepRatio:0.7}
        session.run(op,feed_dict={x:batchXs,y:batchYs,keepRatio:0.7})
        feeds = {x: batchXs, y: batchYs, keepRatio: 1.}
        avg_cost+=session.run(cost,feed_dict=feeds)/total_batch#这里我现在存在疑问对于除数total_batch

    #训练显示
    if epoch%display_step==0:
        print("Epoch:%03d/%03d cost:%9f"%(epoch,epochs,avg_cost))
        train_acc=session.run(accr,feed_dict=feeds)
        print("Trainning accuracy:%.3f "%(train_acc))
        test_acc=session.run(accr,feed_dict={x:mnist.test.images,y:mnist.test.labels,keepRatio:1})
        print("Test accuracy:%.3f "%(test_acc))
print('程序结束')