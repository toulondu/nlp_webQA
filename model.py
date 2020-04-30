# -*- encoding: utf-8 -*-
'''
@FILE           :model.py
@TIME           :2020/04/30 23:32:58
@AUTHOR         :Toulon Du
@EMAIL          :seaduhe@gmail.com
@DESCRIPTION    :
@VERSION        :1.0
'''
import numpy as np
import warnings
import re
import json
import tensorflow as tf
import math
import os
import codecs
import random
import tensorflow.keras.backend as tk

from tqdm import tqdm
from gensim.models import Word2Vec
from tensorflow.keras import Model,Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Lambda,Dense,Dropout,Layer,Conv1D,Softmax,Conv3D

warnings.filterwarnings("ignore")

MAX_LENGTH = 256
HIDDEN_SIZE = 768

WEBQA_DATASET_PATH = './datasets/SogouQA.json'
sougouQA_DATASET_PATH = './datasets/SogouQA.json'

webQA_dataset = json.load(open(WEBQA_DATASET_PATH, encoding='utf-8'))
sougouQA_dataset = json.load(open(sougouQA_DATASET_PATH, encoding='utf-8'))

data_len = len(sougouQA_dataset)
shuffle_order = list(range(data_len))
np.random.shuffle(shuffle_order)
shuffled_sogou_data = [sougouQA_dataset[order] for order in shuffle_order]

dev_dataset = shuffled_sogou_data[0:data_len//3]
train_dataset = shuffled_sogou_data[data_len//3:-1]
# repeat
train_dataset.extend(train_dataset)
# mixed with webQA
train_dataset.extend(webQA_dataset)

word2vec = Word2Vec.load('./wordVecs/word2vec_baike')

# 创建 id2word和word2id dict，id从1开始方便计算，并将word2vec也进行id的对应处理
id2word = {i+1:j for i,j in enumerate(word2vec.wv.index2word)}
word2id = {j:i for i,j in id2word.items()}
word2vec = word2vec.wv.syn0 # 取词向量矩阵
word_size = word2vec.shape[1]
word2vec = np.concatenate([np.zeros((1, word_size)), word2vec]) 


# 对arr数组进行填充并转换为矩阵
def padding_to_max(arr, max_len=MAX_LENGTH, padding = 0):
    if max_len is None:
        len_arr = [len(i) for i in arr]
        max_len = max(len_arr)
    
    return np.array([
        np.concatenate([i, [padding] * (max_len - len(i))]) if len(i) < max_len else i[0:max_len] for i in arr
    ])

def random_truncate_str(str_in,max_len=MAX_LENGTH):
    str_len = len(str_in)
    if str_len <= max_len:
        return str_in
    rand_start = random.randint(0,str_len-max_len)
    return str_in[rand_start:rand_start + max_len]

class data_handler(object):
    def __init__(self,data,batch_size=128):
        self.data = data
        self.batch_size = batch_size
        self.batch_num = math.ceil(len(data) / batch_size)
    
    def __len__(self):
        return self.batch_num

    def __iter__(self):
        while True:
            shuffle_order = list(range(len(self.data)))
            np.random.shuffle(shuffle_order)
            
            # 篇章内容 字id
            x_data_ids = []
            # 篇章字向量
            x_data = []
            # 问题内容 字id
            question_ids = []
            # 问题字向量
            q_data= []
            # 答案位于篇章中的开始和结束标记
            y_label1 = []
            y_label2 = []

            for idx in shuffle_order:
                item = self.data[idx]
                question_ids.append([])
                ques_text = random_truncate_str(item['question'])

                for word in ques_text:
                    question_ids[-1].append(word2id.get(word,0))

                passages_ori = item['passages']
                pi = np.random.choice(len(passages_ori))
                passage = passages_ori[pi]
                pa_cont = random_truncate_str(passage['passage'])
                ans = passage['answer']
                
                x_data_ids.append([])
                for word in pa_cont:
                    x_data_ids[-1].append(word2id.get(word,0))   

                label1,label2 = np.zeros(MAX_LENGTH),np.zeros(MAX_LENGTH)  # 分别用于标识答案的开始位置和结束位置

                if ans:
                    for j in re.finditer(re.escape(ans), pa_cont):
                        label1[j.start()] = 1
                        label2[j.end() - 1] = 1

                y_label1.append(label1)
                y_label2.append(label2)

                if len(x_data_ids) == self.batch_size or idx == shuffle_order[-1]:  
                    x_data_ids = padding_to_max(x_data_ids)
                    question_ids = padding_to_max(question_ids)
                    # 篇章和问题的字向量
                    x_data = word2vec[padding_to_max(x_data_ids,max_len=MAX_LENGTH)]
                    q_data = word2vec[padding_to_max(question_ids,max_len=MAX_LENGTH)]
                    y_label1 = np.array(y_label1)
                    y_label2 = np.array(y_label2)

                    # yield ({'X_ids':x_data_ids,
                    #         'Q_ids':question_ids,
                    #         'X_in':x_data,
                    #         'Q_in':q_data,
                    #         'Y1_in':y_label1,
                    #         'Y2_in':y_label2 }, None)
                    yield [x_data_ids,question_ids,x_data,q_data,y_label1,y_label2],None
                    x_data_ids,question_ids,x_data,q_data,y_label1,y_label2 = [],[],[],[],[],[]

# 自定义层：带门结构的一维卷积 # todo activation 会报错：Could not interpret activation function identifier
class Conv1DWithGate(Layer):
    def __init__(self,
                filters = None,
                kernel_size = 3,
                dilation_rate = 1,
                strides = 1,
                padding = 'same',
                dropout_rate = None,
                activation = 'relu',
                cut_dim = False
                ):
        super(Conv1DWithGate,self).__init__()
        self.filters = filters
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.dropout_rate =  dropout_rate
        self.cut_dim = cut_dim
    
    def build(self,input_shape):
        if self.filters is None:
            self.filters = input_shape[-1]
        self.conv1d = Conv1D(
            self.filters,
            self.kernel_size,
            strides = self.strides,
            dilation_rate = self.dilation_rate,
            padding = self.padding
        )
        self.dropout = Dropout(self.dropout_rate)
        self.softmax = Softmax()

    def call(self,inputs):
        x = inputs
        gate = self.conv1d(x)
        x_conv = self.conv1d(x)

        # 训练阶段对gate进行dropout
        if self.dropout_rate is not None:
            gate = tk.in_train_phase(self.dropout(gate),gate)
        gate = self.softmax(gate)
        
        if self.cut_dim:
            return x
        else:
            # 参考GRU的门结构
            x = x * (1-gate) + x_conv * gate
            return x

# reshape矩阵为2维，最后一个维度不变
def reshape_to_2D(tensor):
    tensor_shape = tensor.shape
    if len(tensor_shape) !=2 :
        return tf.reshape(tensor,[-1,tensor_shape[-1]])

def reshape_to_2D_with_dim_first(tensor):
    tensor_shape = tensor.shape
    if(len(tensor_shape)!=2):
        return tf.reshape(tensor_shape[0],-1)

# 创建一个用于计算attention的mask，from_tensor是输入的句子的矩阵[batch_size, from_seq_length]，
# to_mask为对应的句子的mask，每个句子有词的部分为1，padding的部分为0
# 输出一个[batch_size, from_seq_length, to_seq_length]的矩阵，对应词与词的attention(此处还未计算)，所以是attention_mask
def create_attention_mask_from_input_mask(from_tensor, to_mask,attention_name):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    attention_name: tf.ones 重复调用会报一个 ValueError: Duplicate node name in graph: 'ones/packed' 错误，遂加一个name
  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = tf.shape(from_tensor)
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = tf.shape(to_mask)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.expand_dims(to_mask, 1), tf.float32)

  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32, name=attention_name)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask

# 为方便attention计算,将张量进行reshape和transpose
def transpose_and_reshape_to_cal(input_tensor, batch_size, seq_length, num_heads, head_size):
    """
    对输入张量进行变幻，输出[batch_size, num_attention_heads， seq_length, width]
    """
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_heads, head_size])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

# attention层，用于计算from_tensor和to_tensor的多头attention
def attention_layer(from_tensor,
                to_tensor,
                attention_mask=None,
                num_heads=1,
                size_per_head=128,
                attention_drop_rate = 0.0,
                initializer_range=0.02):
    """attention层，用于计算from_tensor和to_tensor的多头attention
        实现基本参考bert论文和google bert实现，稍作简化
        from_tensor与to_tensor相同则为self-attention

    Args:
        from_tensor: float张量，shape为 [batch_size, 词/字个数seq_length，词向量embedding_size]
        to_tensor: float张量，shape为 [batch_size, 词/字个数seq_length，词向量embedding_size]
        attention_mask: 形状为[batch_size, 词/字个数seq_length, 词/字个数seq_length],两个句子词与词彼此对应关系矩阵，有词对应词的位置为1，否则为0
        num_heads: 多头数量
        size_per_head: 每个head的大小

    Returns:
        float张量，形状为[batch_size,from_tensor的词/字个数seq_length,num_heads*size_per_head]
    """
    from_shape,to_shape = tf.shape(from_tensor),tf.shape(to_tensor)
    if(len(from_shape)!=3 or len(to_shape)!=3):
        raise ValueError('Wrong shape of the input tensor,should be 3d matrix')
    batch_size,from_seq_lenth = from_shape[0],from_shape[1]
    to_seq_length = to_shape[1]

    # 按bert实现，方便表示用：
    # B = batch_size
    # F = `from_tensor` sequence length
    # T = `to_tensor` sequence length
    # N = `num_heads`
    # H = `size_per_head`
    from_2d = reshape_to_2D(from_tensor)  #[B*F,embedding_size]
    to_2d = reshape_to_2D(to_tensor)  #[B*T,embedding_size]
    
    # 下面通过3个FC计算出attention需要的Q，K，V
    # Q：[B*F,N*H]
    Q = Dense(
        num_heads * size_per_head,
        initializer = tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)
    )(from_2d)
    
    # K:[B*T,N*H]
    K = Dense(
        num_heads * size_per_head,
        initializer = tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)
    )(to_2d)

    # V:[B*T,N*H]
    K = Dense(
        num_heads * size_per_head,
        initializer = tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)
    )(to_2d)

    # Q:[B,N,F,H] K:[B,N,T,H]  -> attention(Q,K):[B,N,F,T]
    Q = transpose_and_reshape_to_cal(Q, batch_size, from_seq_lenth, num_heads, size_per_head)
    K = transpose_and_reshape_to_cal(K, batch_size, to_seq_length, num_heads, size_per_head)
    attention_QK = tf.multiply(tf.matmul(Q,K), 1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

         # 对应attention_mask 把词与词对应的位置设置为0，其它设置为-10000
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # 通过广播加到刚才计算好的attention矩阵上
        attention_QK += adder

    attention_softmax = Softmax()(attention_mask)
    attention_softmax = Dropout(rate=attention_drop_rate)(attention_mask)
    
    # V: [B,N,T,H]
    V = transpose_and_reshape_to_cal(V,batch_size,to_seq_length,num_heads,size_per_head)

    # [B,N,F,H]
    attention_res = tf.matmul(attention_QK,V)

    # [B,F,N,H]
    attention_res = tf.transpose(attention_res, [0,2,1,3])

    attention_res = tf.reshape(
        attention_res,
        [batch_size, from_seq_lenth, num_heads*size_per_head]
    )

    return attention_res

class AttentionLayer(Layer):
    def __init__(self,
                num_heads=1,
                size_per_head=128,
                attention_drop_rate = 0.0,
                initializer_range=0.02):
        super(AttentionLayer,self).__init__()
        self.num_heads = num_heads
        self.size_per_head = size_per_head
        self.attention_drop_rate = attention_drop_rate

        self.dense_q = Dense(
            self.num_heads * self.size_per_head,
            kernel_initializer = tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)
        )
        self.dense_k = Dense(
            self.num_heads * self.size_per_head,
            kernel_initializer = tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)
        )
        self.dense_v = Dense(
            self.num_heads * self.size_per_head,
            kernel_initializer = tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)
        )
        
    
    def __call__(self,inputs,mask=None,attention_name=None):
        from_tensor,to_tensor=inputs[0],inputs[1]
        from_mask,to_mask,attention_mask = None,None,None
        if mask is not None:
            from_mask,to_mask = mask
            attention_mask = create_attention_mask_from_input_mask(from_mask,to_mask,attention_name)

        from_shape,to_shape = tf.shape(from_tensor),tf.shape(to_tensor)
        if(from_shape.shape!=3 or to_shape.shape!=3):
            raise ValueError('Wrong shape of the input tensor,should be 3d matrix')
        batch_size,from_seq_lenth = from_shape[0],from_shape[1]
        to_seq_length = to_shape[1]

        # 按bert实现，方便表示用：
        # B = batch_size
        # F = `from_tensor` sequence length
        # T = `to_tensor` sequence length
        # N = `num_heads`
        # H = `size_per_head`
        from_2d = reshape_to_2D(from_tensor)  #[B*F,embedding_size]
        to_2d = reshape_to_2D(to_tensor)  #[B*T,embedding_size]
        
        # 下面通过3个FC计算出attention需要的Q，K，V
        # Q：[B*F,N*H]
        Q = self.dense_q(from_2d)
        
        # K:[B*T,N*H]
        K = self.dense_k(to_2d)

        # V:[B*T,N*H]
        V = self.dense_v(to_2d)

        # Q:[B,N,F,H] K:[B,N,T,H]  -> attention(Q,K):[B,N,F,T]
        Q = transpose_and_reshape_to_cal(Q, batch_size, from_seq_lenth, self.num_heads, self.size_per_head)
        K = transpose_and_reshape_to_cal(K, batch_size, to_seq_length, self.num_heads, self.size_per_head)
        attention_QK = tf.multiply(tf.matmul(Q, K, transpose_b=True), 1.0 / math.sqrt(float(self.size_per_head)))

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            # 对应attention_mask 把词与词对应的位置设置为0，其它设置为-10000
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

            # 通过广播加到刚才计算好的attention矩阵上
            attention_QK += adder

        attention_softmax = Softmax()(attention_mask)
        attention_softmax = Dropout(rate=self.attention_drop_rate)(attention_mask)
        
        # V: [B,N,T,H]
        V = transpose_and_reshape_to_cal(V,batch_size,to_seq_length,self.num_heads,self.size_per_head)

        # [B,N,F,H]
        attention_res = tf.matmul(attention_QK,V)

        # [B,F,N,H]
        attention_res = tf.transpose(attention_res, [0,2,1,3])

        attention_res = tf.reshape(
            attention_res,
            [batch_size, from_seq_lenth, self.num_heads*self.size_per_head]
        )

        return attention_res

def MyModel():
    X_ids_in = Input(shape=(MAX_LENGTH,)) # 篇章id输入
    Q_ids_in = Input(shape=(None,)) # 问题的id输入
    X_in = Input(shape=(None, word_size)) # 篇章词向量输入
    Q_in = Input(shape=(None, word_size)) # 问题的词向量输入
    Y1_in = Input(shape=(None,)) # 答案左边界输入
    Y2_in = Input(shape=(None,)) # 答案右边界输入

    X_ids, Q_ids, X, Q, Y1,Y2 = X_ids_in, Q_ids_in, X_in, Q_in, Y1_in, Y2_in
    X_mask = tf.cast(X_ids>0, 'float32')
    Q_mask = tf.cast(Q_ids>0, 'float32')

    # X_1:[B,MAX_LENGTH,128]  Q_1:[B,MAX_LENGTH,128]
    X_1 = AttentionLayer(size_per_head=128, attention_drop_rate=0.1)([X, X], mask=[X_ids, X_ids], attention_name='self_X')
    Q_1 = AttentionLayer(size_per_head=128, attention_drop_rate=0.1)([Q, Q], mask=[Q_ids, Q_ids], attention_name='self_Q')

    # X_1:[B,MAX_LENGTH,128]
    X_1 = AttentionLayer(size_per_head=128, attention_drop_rate=0.1)([Q_1, X_1], mask=[Q_ids, X_ids], attention_name='Q_X')

    X_1 = Dense(
            word_size,
            kernel_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            activation = 'softmax'
            )(X)

    # [batch_size,MAX_LENGTH,word_size]
    X_1 = Conv1DWithGate(dropout_rate=0.1)(X_1)
    X_1 = Conv1DWithGate(dropout_rate=0.1, dilation_rate=2)(X_1)
    X_1 = Conv1DWithGate(dropout_rate=0.1, dilation_rate=4)(X_1)
    X_1 = Conv1DWithGate(dropout_rate=0.1, dilation_rate=8)(X_1)
    
    X_1 = X_1 + X

    X_1 = Conv1DWithGate(dropout_rate=0.1)(X_1)
    X_1 = Conv1DWithGate(dropout_rate=0.1, dilation_rate=2)(X_1)
    X_1 = Conv1DWithGate(dropout_rate=0.1, dilation_rate=4)(X_1)
    X_1 = Conv1DWithGate(dropout_rate=0.1, dilation_rate=8)(X_1)
    X_1 = Conv1DWithGate(dropout_rate=0.1)(X_1)

    X_1 = X_1 + X

    # X_1 = Conv1DWithGate(X_1.shape[-1]//2, dropout_rate=0.1, cut_dim = True)(X_1)
    # X_1 = Conv1DWithGate(X_1.shape[-1]//2, dropout_rate=0.1, cut_dim = True)(X_1)

    X_1 = Dense(1,activation='softmax')(X_1)

    X_1 = tf.reduce_sum(X_1,X_1.shape[-1])
    
    Y_HAT_START = Dense(X_ids.shape[1])(X_1)
    Y_HAT_START = Softmax()(Y_HAT_START * X_mask)
    Y_HAT_END = Dense(X_ids.shape[1])(X_1)
    Y_HAT_END = Softmax()(Y_HAT_END * X_mask)

    model = tf.keras.Model(inputs=[X_ids_in ,Q_ids_in, X_in, Q_in, Y1_in, Y2_in], outputs=[Y_HAT_START, Y_HAT_END],name='my_model')
    model.summary()
    # loss function 有问题
    # loss1 = focal_loss(Y1_in, Y_HAT_START)
    # # loss1 = tf.keras.losses.MSE(Y1_in, Y_HAT_START)
    # loss1 = tk.sum(loss1 * X_mask) / tk.sum(X_mask)
    # loss2 = focal_loss(Y2_in, Y_HAT_END)
    # # loss2 = tf.keras.losses.MSE(Y2_in, Y_HAT_END)
    # loss2 = tk.sum(loss2 * X_mask) / tk.sum(X_mask)
    # loss = loss1 + loss2 # 放大100倍，可读性好些，不影响Adam的优化

    model.add_loss(loss_function([Y1_in,Y2_in],[Y_HAT_START,Y_HAT_END]))

    return model

def crossentroy(y_true,y_pred):
    return - y_true * tk.log(y_pred) - (1 - y_true) * tk.log(1 - y_pred)

def loss_function(y_true,y_pred):
    [y1,y2] = y_true
    [y_hat1,y_hat2] = y_pred
    y1,y2,y_hat1,y_hat2 = tf.cast(y1,tf.float64),tf.cast(y2,tf.float64),tf.cast(y_hat1,tf.float64),tf.cast(y_hat2,tf.float64)
    loss1 = focal_loss(y1,y_hat1)
    loss1 = tk.sum(loss1)
    loss2 = focal_loss(y2,y_hat2)
    loss2 = tk.sum(loss2)
    loss = loss1 + loss2
    return loss

# copy苏神loss function 和callbacks
def focal_loss(y_true, y_pred):
    alpha, gamma = 0.25, 2
    y_pred = tf.clip_by_value(y_pred, 1e-8, 1 - 1e-8)
    return - alpha * y_true * tk.log(y_pred) * (1 - y_pred)\
           - (1 - alpha) * (1 - y_true) * tk.log(1 - y_pred) * y_pred

class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """
    def __init__(self, model, momentum=0.9999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [tk.zeros(tf.shape(w)) for w in model.weights]
    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = tk.moving_average_update(w1, w2, self.momentum)
            # self.model.metrics_updates.append(op)
    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = tk.batch_get_value(self.model.weights)
        tk.batch_set_value(zip(self.ema_weights, self.old_weights))
    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = tk.batch_get_value(self.model.weights)
        ema_weights = tk.batch_get_value(self.ema_weights)
        tk.batch_set_value(zip(self.model.weights, ema_weights))
    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        tk.batch_set_value(zip(self.model.weights, self.old_weights))

class Evaluate(Callback):
    def __init__(self,model,dev_data):
        self.metrics = []
        self.best = 0.
        self.stage = 0
        self.model = model
        self.dev_data = dev_data
    def on_epoch_end(self, epoch, logs=None):
        # EMAer.apply_ema_weights()
        acc, f1, final = self.evaluate()
        print(f'acc={acc},f1={f1},final={final}')
        self.metrics.append((epoch, acc, f1, final))
        json.dump(self.metrics, open('train.log', 'w'), indent=4)
        if final > self.best:
            self.best = final
            self.model.save_weights('best_model.weights')
        print('learning rate: %s' % (tk.eval(self.model.optimizer.lr)))
        print('acc: %.4f, f1: %.4f, final: %.4f, best final: %.4f\n' % (acc, f1, final, self.best))
        # EMAer.reset_old_weights()
        if epoch + 1 == 30 or (
            self.stage == 0 and epoch > 15 and
            (final < 0.5 or np.argmax(self.metrics, 0)[3] < len(self.metrics) - 5)
        ):
            """达到30个epoch，或者final开始下降到0.5以下（开始发散），
            或者连续5个epoch都没提升，就降低学习率。
            """
            self.stage = 1
            self.model.load_weights('best_model.weights')
            EMAer.initialize()
            tk.set_value(self.model.optimizer.lr, 1e-4)
            tk.set_value(self.model.optimizer.iterations, 0)
            opt_weights = tk.batch_get_value(self.model.optimizer.weights)
            opt_weights = [w * 0. for w in opt_weights]
            tk.batch_set_value(zip(self.model.optimizer.weights, opt_weights))
    def evaluate(self, threshold=0.1):
        predict(self.model, self.dev_data, 'tmp_result.txt', threshold=threshold)
        acc, f1, final = json.loads(
            os.popen(
                'D:/programFiles/python2.7/python ./evaluate_tool/evaluate.py tmp_result.txt tmp_output.txt'
            ).read().strip()
        )
        return acc, f1, final

def predict(model, data, filename, threshold=0.1):
    with codecs.open(filename, 'w', encoding='utf-8') as f:
        for item in iter(data):
            a = predict_answer(model, item, max_answer_len=10, threshold=threshold)
            if a:
                s = u'%s\t%s\n' % (item['id'], a)
            else:
                s = u'%s\t\n' % (item['id'])
            f.write(s)

def predict_answer(model, question, max_answer_len=10, threshold = 0.1):
    """
    为每个question找出最可能的答案

    Args:
        question：一个question数据，包括问题和下面所有的passage
        max_answer: 答案的最大长度
        threshold: 概率gate
    
    Return:
        str: answer_word
    """
    question_text = question['question']
    passages = [item['passage'] for item in question['passages']]
    x_ids = []   
    q_ids = []

    # 每个句子中，找出所有词语中被预测为answer的概率大于阈值的词，取它们在此句中概率最大值存起来
    ans_prob_each_passage = []
    # 根据词语进行汇总
    ans_prob_all = {}
 
    for passage in passages:
        passage = random_truncate_str(passage)
        
        q_ids.append([])
        for word in question_text:
            q_ids[-1].append(word2id.get(word,0))

        x_ids.append([])
        for word in passage:
             x_ids[-1].append(word2id.get(word,0))   
    
    X = word2vec[padding_to_max(x_ids)]
    Q = word2vec[padding_to_max(q_ids)]
    x_ids = padding_to_max(x_ids)
    q_ids = padding_to_max(q_ids)

    # 得到这个问题下所有篇章的答案预测
    y_hat_start, y_hat_end = model.predict([x_ids, q_ids, X, Q, np.ones(x_ids.shape),np.ones(x_ids.shape)])

    for prob_s,prob_e,pa in zip(y_hat_start, y_hat_end, passages):
        
        # 答案只可能出现在句子存在的部分
        prob_s = prob_s[: max(len(pa),MAX_LENGTH)]
        prob_e = prob_e[: max(len(pa),MAX_LENGTH)]
        
        s_idxs = np.where(prob_s > threshold)[0]
        e_idxs = np.where(prob_e > threshold)[0]

        word_with_prob = {}  
        for s_idx in s_idxs:
            e_range = (e_idxs >= s_idx) & (e_idxs < s_idx + max_answer_len)
            for e_idx in e_idxs[e_range]:
                ans_word = pa[s_idx:e_idx+1]
                prob = prob_s[s_idx] * prob_e[e_idx]
                if prob > word_with_prob.get(ans_word,0.0):
                    word_with_prob[ans_word] = prob
        if word_with_prob:
            ans_prob_each_passage.append(word_with_prob)

    for item in ans_prob_each_passage:
        for word,prob in item.items():
            ans_prob_all[word] = ans_prob_all.get(word,[]) + [prob]

    ans_prob_all = {
        word:(np.array(prob)**2).sum() / (sum(prob) + 1) for word,prob in ans_prob_all.items() # 取平方和作为最终评分
    }
    
    # 选出得分最高的词返回
    ans_predict = None
    if ans_prob_all:
        ans_predict = sorted(ans_prob_all.items(), key = lambda x: x[1])[-1][0] 
    return ans_predict


model = MyModel()
model.compile(optimizer='adam',
                metrics=['accuracy'])

train_data = data_handler(train_dataset)

EMAer = ExponentialMovingAverage(model)
EMAer.inject()

evaluator = Evaluate(model,dev_dataset)

if __name__ == '__main__':
    model.fit(train_data.__iter__(),
                steps_per_epoch = len(train_data),
                epochs=100,
                callbacks=[evaluator]
            )
else:
    model.load_weights('best_model.weights')