from mindspore.dataset import GeneratorDataset #导入GeneratorDataset类
import numpy as np #导入numpy工具包


#加载工业数据集
sensor_num = 23 #传感器数量
horizon = 5 #预测的时间步数
PV_index = [idx for idx in range(9)] #PV变量的索引值范围
OP_index = [idx for idx in range(9,18)] #OP变量的索引值范围
DV_index = [idx for idx in range(18,sensor_num)] #DV变量的索引值范围
data_path = 'data/jl_data_train.csv' #数据文件路径
data = np.loadtxt(data_path, delimiter=',', skiprows=1, usecols=range(1,sensor_num+1)) #读取数据（忽略第1行的标题及第1列的时间戳）

if __name__ == '__main__':
    print('数据形状：{0}，元素类型：{1}'.format(data.shape, data.dtype))

    # 绘制传感器数据
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 101), data[:100, 0])  # 绘制第1个传感器的前100条数据
    plt.show()


#定义生成模型输入输出的generateData函数
def generateData(data, X_len, Y_len, sensor_num):#定义generateData函数
    point_num = data.shape[0] #时间点总数
    sample_num = point_num-X_len-Y_len+1 #生成的总样本数
    X = np.zeros((sample_num, X_len, sensor_num)) #用于保存输入数据
    Y = np.zeros((sample_num, Y_len, sensor_num)) #用于保存对应的输出数据
    for i in range(sample_num): #通过遍历逐一生成输入数据和对应的输出数据
        X[i] = data[i:i+X_len] #前X_len个时间点数据组成输入数据
        Y[i] = data[i+X_len:i+X_len+Y_len]#后Y_len个时间点数据组成输出数据
    return X, Y #返回所生成的模型的输入数据X和输出数据Y


#生成用于工业预测模型的数据集
X_t1, Y_t1 = generateData(data, 30, 1, sensor_num) #生成任务1所用的数据集
X_t2, Y_t2 = generateData(data, 30, horizon, sensor_num) #生成任务2所用的数据集

if __name__ == '__main__':
    print('任务1数据集输入数据形状：{0}，输出数据形状：{1}'.format(X_t1.shape, Y_t1.shape))
    print('任务2数据集输入数据形状：{0}，输出数据形状：{1}'.format(X_t2.shape, Y_t2.shape))


#定义用于划分训练集、验证集、测试集的splitData函数
def splitData(X, Y): #定义splitData函数
    N = X.shape[0] #样本总数
    train_X,train_Y=X[:int(N*0.6)],Y[:int(N*0.6)] #前60%的数据作为训练集
    val_X,val_Y=X[int(N*0.6):int(N*0.8)],Y[int(N*0.6):int(N*0.8)] #中间20%的数据作为验证集
    test_X,test_Y=X[int(N*0.8):],Y[int(N*0.8):] #最后20%的数据作为测试集
    return train_X,train_Y, val_X,val_Y, test_X,test_Y#返回划分好的数据集


#生成训练集、验证集和测试集
train_X_t1, train_Y_t1, val_X_t1, val_Y_t1, test_X_t1, test_Y_t1=splitData(X_t1, Y_t1) #划分任务1的数据集
train_X_t2, train_Y_t2, val_X_t2, val_Y_t2, test_X_t2, test_Y_t2=splitData(X_t2, Y_t2) #划分任务2的数据集

if __name__ == '__main__':

    s = '训练集样本数：{0}，验证集样本数：{1}，测试集样本数：{2}'
    print('任务1'+s.format(train_X_t1.shape[0], val_X_t1.shape[0], test_X_t1.shape[0])) #输出任务1训练集、验证集和测试集的样本数
    print('任务2'+s.format(train_X_t2.shape[0], val_X_t2.shape[0], test_X_t2.shape[0])) #输出任务2训练集、验证集和测试集的样本数


#定义多元时间序列数据集类MultiTimeSeriesDataset
class MultiTimeSeriesDataset(): #定义MultiTimeSeriesDataset类
    def __init__(self, X, Y): #构造方法
        self.X, self.Y = X, Y #设置输入数据和输出数据
    def __len__(self):
        return len(self.X) #获取数据的长度
    def __getitem__(self, index):
        return self.X[index], self.Y[index] #根据索引值为index的数据


#定义用于生成训练集、验证集和测试集的generateMindsporeDataset函数
def generateMindsporeDataset(X, Y, batch_size): #定义generateMindsporeDataset函数
    dataset = MultiTimeSeriesDataset(X.astype(np.float32), Y.astype(np.float32)) #根据X和Y创建MultiTimeSeriesDataset类对象
    dataset = GeneratorDataset(dataset, column_names=['data','label']) #创建GeneratorDataset类对象，并指定数据集两列的列名称分别是data和label
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False) #将数据集分成多个批次，以支持批量训练
    return dataset #返回可用于模型训练和测试的数据集


#生成任务1的训练集、验证集和测试集
train_dataset_t1 = generateMindsporeDataset(train_X_t1, train_Y_t1, batch_size=32)
val_dataset_t1 = generateMindsporeDataset(val_X_t1, val_Y_t1, batch_size=32)
test_dataset_t1 = generateMindsporeDataset(test_X_t1, test_Y_t1, batch_size=32)
if __name__ == '__main__':
    for data, label in train_dataset_t1.create_tuple_iterator():
        print('数据形状：', data.shape, '，数据类型：', data.dtype)
        print('标签形状：', label.shape, '，数据类型：', label.dtype)
        break


#生成任务2的训练集、验证集和测试集
train_dataset_t2 = generateMindsporeDataset(train_X_t2, train_Y_t2, batch_size=32)
val_dataset_t2 = generateMindsporeDataset(val_X_t2, val_Y_t2, batch_size=32)
test_dataset_t2 = generateMindsporeDataset(test_X_t2, test_Y_t2, batch_size=32)
