#CIFAR-10
from tensorflow.examples.tutorials.cifar10 import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time

max_steps = 3000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'
#CIFAR-10数据下载默认路径
sess = tf.InteractiveSession()
def variable_with_weight_loss(shape, stddev, wl, name):
	var = tf.Variable(tf.truncated_normal(shape, stddev = stddev), name = name)
	if wl is not None:
		weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name = 'weight_loss')
		tf.add_to_collection('losses', weight_loss)
	return var
#初始化weight函数（附加L2的loss（L2正则化））
#L1正则：制造稀疏特征，大部分无用特征权重置0
#L2正则：特征权重不过大，特征权重比较平均
#奥卡姆剃刀法则

cifar10.maybe_download_and_extract()
#下载数据并解压展开
with tf.name_scope('Get'):
	images_train, labels_train = cifar10_input.distorted_inputs(data_dir = data_dir, batch_size = batch_size)
	'''
	distorted_inputs
	产生训练需要使用的数据（特征，label）
	进行数据增强操作（图片随机水平旋转、随机剪切、随机设置亮度和对比度、数据标准化）
	'''

	images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir, batch_size = batch_size)
	#inputs：生成测试数据（只需进行24X24裁剪+数据标准化）
with tf.name_scope('Inputs'):
	image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3], name = 'image_input')
	label_holder = tf.placeholder(tf.int32, [batch_size], name = 'label_input')
'''
创建输入数据的placeholder（特征+label）
（batch_size后面定义网格结构时用到->样本条数需预先设定）
剪裁后图片大小24X24
颜色通道3（RGB）
'''
with tf.name_scope('Conv1'):
	weight1 = variable_with_weight_loss(shape = [5, 5, 3, 64], stddev = 5e-2, wl = 0.0, name = 'W1')
	with tf.name_scope('kernel1'):
		kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding = 'SAME', name = 'kernel1')
	bias1 = tf.Variable(tf.constant(0.0, shape = [64]), name = 'b1')
	with tf.name_scope('conv1'):
		conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1), name = 'conv1')
	with tf.name_scope('pool1'):
		pool1 = tf.nn.max_pool(conv1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool1')
	with tf.name_scope('norm1'):
		norm1 = tf.nn.lrn(pool1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75, name = 'norm1')
'''
第一个卷积层
卷积核5X5
channel3
卷积核64
标准差0.05
（wl：0->不进行L2正则）
结果进行lrn处理（适合ReLU（ReLU无上限边界）（从附近多个卷积核的相应中挑出比较大的反馈）（不会和Sigmoid（有固有边界+抑制过大值））
'''
with tf.name_scope('Conv2'):
	weight2 = variable_with_weight_loss(shape = [5, 5, 64, 64], stddev = 5e-2, wl = 0.0, name = 'W2')
	with tf.name_scope('kernel2'):
		kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding = 'SAME', name = 'kernel2')
	bias2 = tf.Variable(tf.constant(0.1, shape = [64]), name = 'b2')
	with tf.name_scope('conv2'):
		conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2), name ='conv2')
	with tf.name_scope('norm2'):
		norm2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75, name = 'norm2')
	with tf.name_scope('pool2'):
		pool2 = tf.nn.max_pool(norm2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool2')
'''
第二个卷积层
输入纬度64
bias初始化0.1
先进行lrn在进行最大池化
'''

with tf.name_scope('FC1'):
	reshape = tf.reshape(pool2, [batch_size, -1], name = 'reshape')
	dim = reshape.get_shape()[1].value
	weight3 = variable_with_weight_loss(shape = [dim, 384], stddev = 0.04, wl = 0.004, name = 'W3')
	bias3 = tf.Variable(tf.constant(0.1, shape = [384]), name = 'b3')
	with tf.name_scope('l3'):
		local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3, name = 'l3')
'''
全连接层
get_shape：获取数据扁平化后的长度
（不过拟合->设置weight loss=0.04）
ReLU
'''
with tf.name_scope('FC2'):
	weight4 = variable_with_weight_loss(shape = [384, 192], stddev = 0.04, wl = 0.004, name = 'W4')
	bias4 = tf.Variable(tf.constant(0.1, shape = [192]), 'b4')
	with tf.name_scope('l4'):
		local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4, name = 'l4')
'''
全连接层（削减隐含结点384->192）
'''
with tf.name_scope('Logits'):
	weight5 = variable_with_weight_loss(shape = [192, 10], stddev = 1 / 192.0, wl = 0.0, name = 'W5')
	bias5 = tf.Variable(tf.constant(0.0, shape = [10]), name = 'b5')
	with tf.name_scope('logits'):
		logits = tf.add(tf.matmul(local4, weight5), bias5, name = 'logits')
'''
最终层
'''

'''
卷积神经网络结构
conv1：卷积层和ReLU激活函数
->pool1：最大池化
->norm1：LRN
->conv2：卷积层和ReLU激活函数
->norm2：LRN
->pool2：最大池化
->local3：全连接层和ReLU激活函数
->local4：全连接层和ReLU激活函数
->logits：模型Inference的输出结果
'''

def loss(logits, labels):
	labels = tf.cast(labels, tf.int64, name = 'labels')
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels, name = 'cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	return tf.add_n(tf.get_collection('losses'), name = 'total_loss')
'''
cross_entropy：tf.nn.sparse_softmax_cross_entropy_with_logits合并softmax计算和cross_entropy loss计算
cross_entropy_mean：计算cross_entropy均值
将cross_entropy的loss整合到loss中得到最终的loss
一共包括cross_entropy的loss和全连接层weight的L2 loss
'''
with tf.name_scope('loss'):
	loss = loss(logits, label_holder)
	tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
	train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

with tf.name_scope('test'):
	top_k_op = tf.nn.in_top_k(logits, label_holder, 1, name = 'top_one')

'''
tf.nn.in_top_k：输出结果中top k的准确率（默认使用top 1（默认输出分数最高的那一类的准确率））
'''



merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/CNN_pro/', tf.get_default_graph())

tf.global_variables_initializer().run()

tf.train.start_queue_runners()
#启动图片数据增强的线程队列

for step in range(max_steps):
	start_time = time.time()
	image_batch, label_batch = sess.run([images_train, labels_train])
	#sess.run：获取训练batch数据
	temp, loss_value = sess.run([train_op, loss], feed_dict = {image_holder: image_batch, label_holder: label_batch})
	duration = time.time() - start_time
	if step % 10 == 0:
		examples_per_sec = batch_size / duration
		sec_per_batch = float(duration)

		format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
		result = sess.run(merged, {image_holder: image_batch, label_holder: label_batch})
		writer.add_summary(result, step)
		print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

num_examples = 10000
#测试集10000个样本
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
	image_batch, label_batch = sess.run([images_test, labels_test])
	predictions = sess.run([top_k_op], feed_dict = {image_holder: image_batch, label_holder: label_batch})
	true_count += np.sum(predictions)
	step += 1

precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)

input()
