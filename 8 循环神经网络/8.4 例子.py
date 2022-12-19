import tensorflow as tf
import numpy as np
import matplotlib as mpl

#运行不显示图
mpl.use('Agg')
from matplotlib import pyplot as plt

#隐藏层节点个数
HIDDEN_SIZE=30
#Lstm层数
NUM_LAYERS=2
#循环神经网路的训练序列长度
TIMESTEPS=10
#训练轮数
TRAING_STEPS=10000
BATCH_SIZE=32

#训练数据个数
TRAING_EXAMPLES=10000
#测试数据个数
TESTING_EXAMPLES=1000
#采样间隔
SAMPLE_GAP=0.01

def generate_data(seq):
    x=[]
    y=[]

    #序列的第i项和后面的TIMESTEPS-1项作为输入，
    for i in range(len(seq)-TIMESTEPS):
        x.append([seq[i:i+TIMESTEPS]])
        y.append([seq[i+TIMESTEPS]])
    return np.array(x,dtype=np.float32),np.array(y,dtype=np.float32)


def lstm_model(x,y,is_training):
    #使用多层的LSTM结构
    cell=tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)]
    )


    #使用TensorFlow 接口将多层的LSTM结构链接成RNN网络并计算前向传播结果
    #outputs 是顶层Lstm 在每一步的输出结果，纬度是[batch_size,time,HIDDEN_SIZE],本问题中只关注最后一个时刻的输出结果
    outputs,_=tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
    output=outputs[:,-1,:]

    #对lstm网络的输出加已成全联接并计算损失
    predictions=tf.contrib.layers.fully_connected(output,1,activation_fn=None)

    #只在训练时计算损失函数和优化步骤，测试时直接返回预测结果
    if not is_training:
        return predictions,None,None

    loss=tf.losses.mean_squared_error(labels=y,predictions=predictions)

    #创建模型优化器并得到优化步骤
    train_op=tf.contrib.layers.optimize_loss(
        loss,tf.train.get_global_step(),
        optimizer='Adagrad',learning_rate=0.1
    )

    return predictions,loss,train_op


def train(sess,train_x,train_y):
    #将训练数据以数据集的方式提供给计算图
    ds=tf.data.Dataset.from_tensor_slices((train_x,train_y))
    ds=ds.repeat().shuffle(1000).batch(BATCH_SIZE)

    x,y=ds.make_one_shot_iterator().get_next()

    #调用模型，得到预测结果，损失函数，和训练操作
    with tf.variable_scope('model'):
        predictions,loss,train_op=lstm_model(x,y,True)

        #初始化变量
        sess.run(tf.global_variables_initializer())
        for i in range(TRAING_STEPS):
            _,l=sess.run([train_op,loss])
            if(i%1000==0):
                print("train step:"+str(i)+",loss:"+str(l))

#评估
def run_eval(sess,test_x,test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    ds = ds.batch(1)

    x,y=ds.make_one_shot_iterator().get_next()


    #调用模型得到计算结果，不需要输入真实的y值
    with tf.variable_scope('model',reuse=True):
        prediction,_,_=lstm_model(x,[0.0],False)


    #将预测结果存入一个数组
    predictions=[]
    lables=[]
    for  i in range(TESTING_EXAMPLES):
        p,l=sess.run([predictions,y])
        predictions.append(p)
        lables.append(l)

    #计算rmse作为评价指标
    #squeeze去除纬度为1的shape
    predictions=np.array(predictions).squeeze()
    lables=np.array(lables).squeeze()
    rmse=np.sqrt((predictions-lables)**2).mean(axis=0)
    print("MEAN  SQUARE ERROR is:%f" %rmse)

    #对预测的sin函数作绘图
    plt.figure()
    plt.plot(predictions,labe='predictions')
    plt.plot(lables,lable='real_sin')
    plt.legend()
    plt.show()

#使用正弦函数生成训练集和测试集
test_start=(TRAING_EXAMPLES+TIMESTEPS)*SAMPLE_GAP
test_end=test_start+(TESTING_EXAMPLES+TIMESTEPS)*SAMPLE_GAP

train_x,train_y=generate_data(np.sin(np.linspace(0,test_start,TRAING_EXAMPLES+TIMESTEPS,dtype=np.float32)))
test_x,test_y=generate_data(np.sin(np.linspace(test_start,test_end,TESTING_EXAMPLES+TIMESTEPS,dtype=np.float32)))

with tf.Session() as sess:
    train(sess,train_x,train_y)
    run_eval(sess,test_x,test_y)






