'''
深度学习
	1、无监督学习
	2、逐层抽象
'''
# 稀疏编码
# 自编码器：使用自身的高阶特征编码自己
# 去噪自编码器
# 噪声 常用 AGN加性高斯噪声 Masking Noise随机遮挡噪声
# tf自编码器实现
# 使用无监督的自编码器来提取特征
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# xavier自编码器参数初始化
def xavier_init(fan_in, fan_out, constant=1):
    low = - constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


'''
去噪自编码
n_input 输入变量数
n_hidden隐含层节点数
transfer_function隐含层激活函数 （softplus）
optimizer优化器（Adam）
scale高斯噪声系数（0.1）
'''


class AdditiveGaussianNoiseAutoencoder(object):
    '''
    AGN
    包括:
    神经网络的设计
    权重初始化
    成员函数（transform，generate）（用于计算图中的子图）
    '''

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(),
                 scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        # 定义网络结构 P61
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(
            tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        # 定义自编码器损失函数 P61
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        # 创建字典
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    # 初始化权重和偏置系数

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    # 训练完毕时进行评测使用（不触发训练）

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    # 返回自编码器隐含层的输出结果
    # 隐含层功能：学习出数据中的高阶特征

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    '''
    重建层（自编码器后半部分）：
    输入：隐含层的输出结果
    将提取到的高阶特征复原为原始数据
    '''

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    # 获取隐含层权重（w1）

    def getBiases(self):
        return self.sess.run(self.weights['b1'])


# 获取隐含层偏置系数（b1）


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def standard_scale(X_train, X_test):
    preprossor = prep.StandardScaler().fit(X_train)
    X_train = preprossor.transform(X_train)
    X_test = preprossor.transform(X_test)
    return X_train, X_test


# 训练、测试数据进行标准化处理（0均值，1标准差）
# 训练、测试数据都使用完全相同的scale

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]
    '''
    获取随机block数据的函数:
        取一个从0到len（data）-batch_size之间的随机整数
        以随机数作为block的起始位置，顺序取到一个batch size的数据
        （属于不放回抽样）
    '''


X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
# 对训练集、测试集进行标准化变换

n_samples = int(mnist.train.num_examples)
training_epochs = 20  # 最大训练轮数
batch_size = 128
display_step = 1  # 每隔多少轮显示一次损失cost

autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784, n_hidden=200, transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001), scale=0.01)

# 开始训练
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size

    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost:", "{:.9f}", format(avg_cost))
print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
input()
