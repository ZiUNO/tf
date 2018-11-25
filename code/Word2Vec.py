import collections
import math
import random
import zipfile
import numpy as np
import tensorflow as tf
import os
import urllib.request

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

url = 'http://mattmahoney.net/dc/'


def maybe_download(file, expected_bytes):
    file_path = os.getcwd() + "\\src\\Word_data\\" + file
    if not os.path.exists(file_path):
        file, _ = urllib.request.urlretrieve(url + file, file_path)
    stat = os.stat(file_path)
    if stat.st_size == expected_bytes:
        print('Found and verified', file)
    else:
        print(stat.st_size)
        raise Exception(
            'Failed to verify ' + file + '. Can you get to it with a browser?')
    return file


filename = maybe_download('text8.zip', 31344016)


def read_data(file):
    with zipfile.ZipFile(os.getcwd() + "\\src\\Word_data\\" + file) as f:
        return tf.compat.as_str(f.read(f.namelist()[0])).split()


words = read_data(filename)
print('Data size', len(words))

vocabulary_size = 50000


def build_data_set(words_data):
    """
    * 创建数据集合
    :param words_data: 单词列表
    :returns:
        word_code_data: 编码后的单词列表 <br />
        counter: 进行计数的单词列表 <br />
        diction: 字典（编码表） <br />
        reverse_diction: 码值反向对应
    """
    counter = [['UNK', -1]]
    counter.extend(collections.Counter(words_data).most_common(vocabulary_size - 1))
    diction = dict()
    for word, _ in counter:
        # TODO 对word进行编码
        diction[word] = len(diction)
    word_code_data = list()
    unk_count = 0
    for word in words_data:
        # TODO 对所有单词进行编码
        if word in diction:
            index = diction[word]
        else:
            index = 0
            unk_count += 1
        word_code_data.append(index)
    counter[0][1] = unk_count
    # NOTE [zip] 迭代打包每一组
    reverse_diction = dict(zip(diction.values(), diction.keys()))
    return word_code_data, counter, diction, reverse_diction


data, count, dictionary, reverse_dictionary = build_data_set(words)

# TODO words 变量释放
del words
# TODO 打印出现最多的前5
print('Most common words (+UNK)', count[:5])
# TODO 打印示例数据：前10个单词的编码+相应的单词
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


def generate_batch(batches_size, num_skip, skips_window):
    """
    * 获取batch数据
    :param batches_size: batch大小
    :param num_skip: 每个单词生成的样本数
    :param skips_window: 单词最远可联系的距离
    :returns:
        batches:  <br />
        label:
    """
    global data_index
    # NOTE [assert] 断言
    assert batches_size % num_skip == 0
    assert num_skip <= 2 * skips_window
    batches = np.ndarray(shape=batches_size, dtype=np.int32)
    label = np.ndarray(shape=(batches_size, 1), dtype=np.int32)
    # NOTE [span] 为某个单词创建相关样本时会用到的单词数量（单词本身+前后单词）
    span = 2 * skips_window + 1
    # NOTE [buffer] 大小为span的双向队列
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batches_size // num_skip):
        target = skips_window
        targets_to_avoid = [skips_window]
        for j in range(num_skip):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batches[i * num_skip + j] = buffer[skips_window]
            label[i * num_skip + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batches, label


batch, labels = generate_batch(batches_size=8, num_skip=2, skips_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0])

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_data_set = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(
        weights=nce_weights,
        biases=nce_biases,
        labels=train_labels,
        inputs=embed,
        num_sampled=num_sampled,
        num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalizer_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalizer_embeddings, valid_data_set)
    similarity = tf.matmul(valid_embeddings, normalizer_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()

num_steps = 100001

with tf.Session(graph=graph) as session:
    init.run()
    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print('Average loss at step ', step, ':', average_loss)
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            nearest = 0
            top_k = 8
            log_str = 0
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
    final_embeddings = normalizer_embeddings.eval()
