#卷积神经网路
#https://blog.csdn.net/u012871279/article/details/78036503
#CNN
'''
图像识别和视频问题
时间序列信号（音频信号、文本数据）
'''
'''
卷积层操作：
1、图像通过多个不同的卷积核滤波，加偏置，提取局部特征（每个卷积核映射出一个新的2D图像）
2、滤波结果进行非线性激活函数处理（ReLU）
3、对激活函数结果进行池化（即降采样），最大池化（保留最显著的特征）
（可附加LRN层（局部响应归一化层）（Trick、Batch Normalization））
'''
'''
卷积神经网络：（要点）
1、局部连接（降低参数量、减轻过拟合）
2、权值共享（降低参数量、减轻过拟合、对平移的容忍性）
3、池化层中的降采样（进一步降低输出参数量，对轻微形变的容忍性）
'''
'''
LetNet5特性：
1、每个卷积层包含三部分：卷积、池化、非线性激活函数
2、使用卷积提取空间特征
3、降采样的平均池化层
4、双曲正切或S型的激活函数
5、MLP作为最后的分类器
6、层与层之间的稀疏连接减少计算复杂度
结构：
	1、输入
	2、三个卷积层
	3、全连接层
	4、高斯连接层
'''
'''
letNet5结构：
1、C1：卷积，6个卷积核，5*5
2、S2：池化，平均池化，2*2（Sigmoid激活函数：进行非线性处理）
3、C3：卷积，16个卷积核（只连接部分Feature Map），5*5
4、S4：池化，平均池化，2*2
5、C5：卷积，120个卷积核，5*5
6、F6：全连接层（Sigmoid激活函数）
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

def weight_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name = name)
	#截断的正态分布，标准差0.1
	'''
	shape（[a, b, c, d])
	a,b：卷积核大小
	c：channel
	d：卷积核个数
	'''

def bias_variable(shape, name):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name = name)
	#（使用ReLU）偏置初始为正值0.1（避免死亡节点）
#定义初始化函数以便复用

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
	'''
	tf.nn.conv2d：tf中的2维卷积函数
	x：参数
	W：卷积参数
	（例：[5, 5, 1, 32] 
	前两个数字：卷积核大小 
	第三个数字：channel个数（由图片种类决定（灰度单色->1， RGB->3）
	最后一个数字：卷积核个数
	strides：卷积核模板移动的步长
	padding：边界处理方式（'SAME'：边界+Padding->卷积输出和输入保持同样尺寸
	'''

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
	'''
	最大池化
	2X2->1X1
	缩小图片尺寸
	参数：
		x：value，池化输入（feature map）（shape为[batch, height, width, channels])
		ksize：池化窗口大小，一般为[1, height, width, 1]（batch和channels不进行池化，维度设为1）
		strides：卷积核在每个维度的步长
	'''

#定义卷积层和池化层（复用）
with tf.name_scope('Inputs'):
	x = tf.placeholder(tf.float32, [None, 784], name = 'x_input')
	y_ = tf.placeholder(tf.float32, [None, 10], name = 'y_input')
	with tf.name_scope('x_image'):
		x_image = tf.reshape(x, [-1, 28, 28, 1], name = 'x_image')
'''
x：1D->x_image：2D（1*784->28*28）[转换为原始的28*28结构]
tf.reshape
参数
	x：输入
	[-1, 28, 28, 1]
	batch（-1）：样本数量不固定
	channel（1）：1个颜色通道
'''
with tf.name_scope('Conv1'):
	W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
	b_conv1 = bias_variable([32], 'b_conv1')
	with tf.name_scope('conv_1'):
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	with tf.name_scope('pool1'):
		h_pool1 = max_pool_2x2(h_conv1)
'''
第一层卷积层
卷积核5X5,通道1，卷积核个数32
'''
with tf.name_scope('Conv2'):
	W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
	b_conv2 = bias_variable([64], 'b_conv2')
	with tf.name_scope('conv_2'):
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	with tf.name_scope('pool2'):
		h_pool2 = max_pool_2x2(h_conv2)
'''
第二层卷积层
卷积层5X5，通道32，卷积核个数64
'''

with tf.name_scope('FC'):
	W_fc1 = weight_variable([7 * 7 * 64, 1024], 'W_fc')
	b_fc1 = bias_variable([1024], 'b_fc')
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name = 'pool2_1D')
	with tf.name_scope('fc'):
		h_fc1 =tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name = 'activation')
'''
全连接层
28*28 / 4*4 = 7*7
64层
2D->1D（7*7*64->1*n）
1024个隐含结点
激活函数
'''

with tf.name_scope('Dropout'):
	keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name = 'keep_fc')
'''
Dropout层
'''

with tf.name_scope('Softmax'):
	W_fc2 = weight_variable([1024, 10], 'W_softmax')
	b_fc2 = bias_variable([10], 'b_sotfmax')
	with tf.name_scope('softmax'):
		y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name = 'y')
'''
Softmax层（->概率）
Dropout层输出
'''
with tf.name_scope('loss'):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices = [1]))
	tf.summary.scalar('cross_entropy',cross_entropy)
with tf.name_scope('train'):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#定义loss和train_step

with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1), name = 'correct_prediction')
        with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
        tf.summary.scalar('accuracy',accuracy)
#定义评测准确性
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/', tf.get_default_graph())

tf.global_variables_initializer().run()
#初始化
for i in range(20001):
	#总训练样本100W
	batch = mnist.train.next_batch(50)
	if i % 100 == 0:
		train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
		#keep_prob=1：实时监测模型性能

		print("step %d, training accuracy %g" % (i, train_accuracy))
	train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})
	if i % 100 == 0:
		result = sess.run(merged, {x: batch[0], y_: batch[1], keep_prob: 1.0})
		writer.add_summary(result, i)
 	#训练Dropout：0.5

mean_value = 0.0
for i in range(mnist.test.labels.shape[0]):
        batch = mnist.test.next_batch(50)
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        mean_value += train_accuracy
 
print("test accuracy %g" % (mean_value / mnist.test.labels.shape[0]))
input()


