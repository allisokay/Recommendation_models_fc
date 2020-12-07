import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
tf=tf.compat.v1
tf.logging.set_verbosity(tf.logging.ERROR)

#读取数据
mnist=input_data.read_data_sets('../dataset/mnist/', one_hot=True)#加载mnist数据集
#print(mnist)
train_img=mnist.train.images   #获取mnist数据集中的训练集中的图片
train_labels=mnist.train.labels  #获取mnist数据集中的训练集中的标签
test_img=mnist.test.images       #获取mnist中的测试集中的图片
test_labels=mnist.test.labels   #获取mnist数据集中的测试集标签
print('Mnist loaded')

#申请变量占位
x=tf.placeholder('float',[None,784])#784是图片维度，none表示行数未知，请自动生成,这里用于加载图片
y=tf.placeholder('float',[None,10])#10是标签的维度

#构建模型初始参数：初始的参数是随机的，怎么取都行
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable([10],dtype=float)

#回归模型：
actv=tf.nn.softmax(tf.matmul(x,W)+b)#比起线性回归不同的是，这里除了y=Wx=b之外，逻辑回归有激活函数
#损失函数loss funciton 或称代价函数 cost function
loss=tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv),reduction_indices=1))
#优化也就是损失，调参使最小化误差
learning_rate=0.01
op=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#预测
pred=tf.equal(tf.argmax(actv,1),tf.argmax(y,1))
#正确率
accr=tf.reduce_mean(tf.cast(pred,'float'))
#初始化
init=tf.global_variables_initializer()
#session
session=tf.Session()
session.run(init)

#每多少次迭代显示一次损失
train_epochs=500
#批尺寸
batch_size=100
#训练迭代次数
display_step=5
for epoch in range(train_epochs):
    avg_loss=5
    #55000/100
    num_batch=int(mnist.train.num_examples/batch_size)
    for i in  range(num_batch):
        #获取数据集,next_batch获取下一批的数据
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        #模型训练
        session.run(op,feed_dict={x:batch_xs,y:batch_ys})
        feeds={x:batch_xs,y:batch_ys}
        avg_loss=session.run(loss,feed_dict=feeds)/num_batch

    if epoch % display_step==0:
        feeds_train={x:batch_xs,y:batch_ys}
        feeds_test={x:mnist.test.images,y:mnist.test.labels}
        train_acc=session.run(accr,feed_dict=feeds_train)
        test_acc=session.run(accr,feed_dict=feeds_test)
        print("Epoch:%03d/%03d cost: %9f train_acc:%3f test_acc:%3f"%(epoch,train_epochs,avg_loss,train_acc,test_acc))

print("DONE")