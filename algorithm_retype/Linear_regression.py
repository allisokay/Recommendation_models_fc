import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# 随机生成一千个点，围绕在y=0.1x+0.3的直线范围内
num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)  # 生成一个均值为0.0，方差为0.55的高斯分布
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)  # 声明一个y=0.1x+0.3的函数，增加抖动（wx+b）
    # 放入向量集中
    vectors_set.append([x1, y1])

# 生成一些样本
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]
# 点的显示

plt.scatter(x_data, y_data, c='r')
plt.show()

'''模型训练:形成线性回归'''
# 生成一维w矩阵，取值范围为【-1,1】的随机数
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
# 生成一个一维的B矩阵，初始值为0
b = tf.Variable(tf.zeros([1]), name='b')
# 经过计算得到预估值y
y = W * x_data + b
# 将预估值y和真实值y_data之间的均方差作为损失
loss = tf.reduce_mean(tf.square(y-y_data), name='loss')
# 使用梯度下降法来优化参数
op = tf.train.GradientDescentOptimizer(0.5)
# 训练的过程就是最小化这个误差值
train = op.minimize(loss, name='train')
# 建立会话
session = tf.Session()
# 全局变量的初始化
init = tf.global_variables_initializer()
# 执行训练
session.run(init)
# 查看初始化的W和b的值
print('W=', session.run(W), 'b=', session.run(b), 'loss=', session.run(loss))

#进行训练30次
for step in range(30):
    session.run(train)
    print('step:'+str(step),'W=', session.run(W), 'b=', session.run(b), 'loss=', session.run(loss))

plt.scatter(x_data, y_data, c='r')
plt.plot(x_data, session.run(W) * x_data + session.run(b))
plt.show()
