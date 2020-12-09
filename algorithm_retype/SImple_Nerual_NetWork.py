import numpy as np
import tensorflow as tf
tf=tf.compat.v1
tf.logging.set_verbosity(tf.logging.ERROR)
import utils.loadMnistDataUtil as mnist

'''加载数据'''
print("-"*30,'数据加载',"-"*30,'\n数据形状:')
train_imgs,train_labs,test_imgs,test_labs,_mnist=mnist.getMnistData()
print("训练图片:",train_imgs.shape,"训练标签:",train_labs.shape)
print("测试图片:",test_imgs.shape,"测试标签",test_labs.shape,'\n',"-"*30,'数据加载完成',"-"*30)

'''网络拓扑结构'''
nHidden1=256 #第一个隐藏层的神经元数
nHidden2=128 #第二个隐藏层的神经元数
nInput=784   #输入像素
nClasses=10  #维度或称标签数
#TensorFlow输入设置
x=tf.placeholder('float',[None,nInput])
y=tf.placeholder('float',[None,nClasses])
#神经网络
stddev=0.1#方差设置
weights={#以字典存储形式存储权重
  "w1":tf.Variable(tf.random_normal([nInput,nHidden1],stddev=stddev)),#注意数据进入网络时，第一层网络的权重行数一定等于数据矩阵的列数
  "w2":tf.Variable(tf.random_normal([nHidden1,nHidden2],stddev=stddev)),
  "out":tf.Variable(tf.random_normal([nHidden2,nClasses],stddev=stddev))#数据映射到输出层网络时，网络权重的列数要等于标签维度
}
biases={
    "b1":tf.Variable(tf.random_normal([nHidden1])),#注意这里是偏置矩阵，它的行数应该对应它相应权重矩阵的行数
    "b2":tf.Variable(tf.random_normal([nHidden2])),
    "out":tf.Variable(tf.random_normal([nClasses]))
}
print("-"*30,'tf搭建神经网络设置',"-"*30)
print("第一层神经网络\n权重w1:",weights.get("w1"),"    偏置b1:",biases.get("b1"))
print("第二层神经网络\n权重w2:",weights.get("w2"),"    偏置b2:",biases.get("b2"))
print("输出层神经网络\n权重wOut",weights.get("out")," 偏置bOut:",biases.get("out"))
#激活函数s,操作数据(这里应该是模型构建，加上输出层有三层构建，前亮层一边构建一边激活)
def multilay_perceotron(_X,_wights,_biases):#BP算法
    #模型构建并激活：第一层：可以看出这是线性的
    modelFuc=tf.add(tf.matmul(_X,_wights.get("w1")),_biases.get("b1"))
    layer1Model=tf.nn.sigmoid(modelFuc)
    # 模型构建并激活：第二层
    modelFuc=tf.add(tf.matmul(layer1Model,_wights.get("w2")),_biases.get("b2"))
    layer1Mode2=tf.nn.sigmoid(modelFuc)
    #模型完善
    _model=tf.add(tf.matmul(layer1Mode2,_wights.get("out")),_biases.get("out"))
    return _model

#获取模型预测结果
pred=multilay_perceotron(x,weights,biases)
#损失函数:交叉熵损失(在线性回归中用的是均方差损失tf.reduce_mean(tf.squre(y-_y)，name='loss')，而且不用激活
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
#优化函数：梯度下降minimize优化,学习率为0.001
op=tf.train.GradientDescentOptimizer(0.001).minimize(cost)
#正确性预测：想等则返回true,不等则返回false
corr=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
#准确率预测。把True转换为1，false转换为0，然后相加
acc=tf.reduce_mean(tf.cast(corr,"float"))#tf.cast将True转换为1.0，false转换为0.0
#初始化
init=tf.global_variables_initializer()
session=tf.Session()
session.run(init)
#模型训练：
epochs=100
batch_size=100
display_step=4#打印次数
for epoch in range(epochs):
    avg_cost=0#平均误差
    total_batch=int(_mnist.train.num_examples/batch_size)#55000/100=1100批
    for i in range(total_batch):
        batchXs,batchYs=_mnist.train.next_batch(batch_size)
        feeds={x:batchXs,y:batchYs}
        session.run(op,feed_dict=feeds)
        avg_cost=session.run(cost,feed_dict=feeds)
    avg_cost=avg_cost/total_batch
    #显示
    if(epoch+1)%display_step==0:
        print("Epoch:%03d/%03d cost:%.9f"%(epoch,epochs,avg_cost))
        feeds={x:batchXs,y:batchYs}
        trainAcc=session.run(acc,feed_dict=feeds)
        print("TRAIN ACCURACY:%.3f"%(trainAcc))
        feeds={x:_mnist.test.images,y:_mnist.test.labels}
        testAcc=session.run(acc,feed_dict=feeds)
        print("TEST ACCURACY:%.3f"%(testAcc))
print("finished")
