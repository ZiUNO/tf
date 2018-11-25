# training set 训练集 60000
# test set 测试集 10000
# 28*28=784
# python 3.6
# 注释均在下方
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
# softmax回归

sess = tf.InteractiveSession()
# 将session注册为默认session（不同session的数据和运算独立）
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, 784], name="x_input")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_input")
'''
placeholder，输入数据的地方
参数1：数据类型 
参数2：tensor的shape（数据尺寸）（None不限条数，784为每条输入为784维）
'''
# wights biases初始化 0
# 卷积网络、循环网络、全连接网络
with tf.name_scope('layer'):
    W = tf.Variable(tf.zeros([784, 10]), name="W")
    tf.summary.histogram('layer/W', W)
    b = tf.Variable(tf.zeros([10]), name="b")
    tf.summary.histogram('layer/b', b)
    y = tf.nn.softmax(tf.matmul(x, W) + b, name="y")
    tf.summary.histogram('layer/y', y)
# loss function 描述模型对问题的分类精度
# （通常使用cross-entropy 损失函数）
# loss初始0
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
# tf.reduce_sum 求和 reduce_mean 对每个batch数据结果求平均值
# 定义优化算法（随机梯度下降 SGD）
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
'''
优化器tf.train.GradientDescentOptimizer 
学习速率 0.5 
优化目标 cross_entropy
'''
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/', tf.get_default_graph())
tf.global_variables_initializer().run()
'''
tf全局参数初始化器tf.global_variables_initializer
run方法
'''
# 执行训练操作 每次执行部分训练样本
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})
    if i % 50 == 0:
        result = sess.run(merged, {x: batch_xs, y_: batch_ys})
        writer.add_summary(result, i)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
'''
tf.argmax 从tensor中寻找最大值的序号
tf.argmax(y,1)求各个预测数字中概率最大的那一个
tf.equal 判断数字类别是否正确
'''
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 精准度
# tf.cast bool转换为float32
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
# 计算准确率
'''
流程：（核心步骤）
	1、定义算法公式（神经网络forward时的计算）
	2、定义loss，选定优化器，并指定优化器优化loss
	3、迭代地对数据进行训练
	4、在测试集或验证集上对准确率进行评测
'''
'''
深度学习
	1、无监督学习
	2、逐层抽象
'''
# 稀疏编码
# 自编码器
# 去噪自编码器
# 噪声 常用 AGN加性高斯噪声 Masking Noise随机遮挡噪声
'''
writer =  tf.summary.FileWriter('/path/to/logs',tf.get_default_graph())
writer.close()
'''
# input()
print("Done!")
input()
