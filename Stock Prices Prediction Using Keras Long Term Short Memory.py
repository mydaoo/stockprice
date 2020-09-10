# Part 1 - Data Preprocessing 数据预处理

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the training set   导入训练集
dataset_train = pd.read_csv('NSE-TATAGLOBAL.csv')   # 打开历史价格文件：塔塔全球饮料有限公司，后期可替换
# df.iloc[:,[0]] #取第0列所有行，多取几列格式为 data.iloc[:,[0,1]]
training_set = dataset_train.iloc[:, 1:2].values  # 模型选择第二和第三列即开盘价(Open)和最高价(High)。 Date为第一列
# df.values返回给定DataFrame的numpy表示形式 也就是Ndarray ： N 维数组对象ndarray
dataset_train.head()   # 我们查看数据集的表头，可以大致了解数据集的类型。

# Feature Scaling   功能缩放
from sklearn.preprocessing import MinMaxScaler  # 使用Scikit- Learn的MinMaxScaler函数将数据集归一到0到1之间
sc = MinMaxScaler(feature_range = (0, 1))        # 数据归一化
training_set_scaled = sc.fit_transform(training_set)    # 设置训练集

# LSTM要求数据有特殊格式，通常是3D数组格式。
# Creating a data structure with 60 timesteps and 1 output   创建具有60个时间步长和1个输出的数据结构
X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])  # 初始按照60的步长创建数据
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)   # 通过Numpy转化到数组中

# Reshaping  将X_train的数据转化到3D维度的数组中，时间步长设置为60，每一步表示一个特征。
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN  构建RNN - 循环神经网络

# Importing the Keras libraries and packages   导入Keras库和包

from keras.models import Sequential
from keras.layers import Dense   # 全连接网络层
from keras.layers import LSTM   # 长短时记忆层（LSTM）
from keras.layers import Dropout   # 添加dropout 随机失活层防止过拟合

# Initialising the RNN   初始化RNN循 环神经网络
regressor = Sequential()   # 回归量 = 序惯模型 函数

# Adding the first LSTM layer and some Dropout regularisation  防止过拟合，添加第一层LSTM层和随机失活层正则化
# 50 units 表示输出空间是50维度的单位，return_sequences=True 表示是返回输出序列中的最后一个输出，还是返回完整序列，input_shape 训练集的大小
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) # 参数为0.2，意味着将删除20%的层

# Adding a second LSTM layer and some Dropout regularisation  重复添加多层网络等于添加隐藏层
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation  添加第三个LSTM层和随机失活层正则化
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation  添加第四个LSTM层和随机失活层正则化
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1)) # 指定1个单元的输出作为全连接层（Dense layer）

# Compiling the RNN  使用目前流行的adam优化器编译模型，并用均方误差（mean_squarred_error）来计算误差
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')   # .compile编译   optimizer 优化器  loss= mse定义损失函数

# Fitting the RNN to the Training set  # 将循环网络安装到训练集上
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)  # 100轮，每批尺寸32

# Part 3 - Making the predictions and visualising the results   在测试集上预测股价并可视化结果

# Getting the real stock price of 2017  获取2017年的真实股价
dataset_test = pd.read_csv('tatatest.csv')   # 导入股价预测的测试集
real_stock_price = dataset_test.iloc[:, 1:2].values    # 提取第二三列并转换为Ndarray数组对象

# Getting the predicted stock price of 2017   获取2017年的预测股价
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)   # 在0轴上合并训练集和测试集
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values  # 将时间步长设置为60(如前面所介绍的)
inputs = inputs.reshape(-1,1)  # 转化为固定只有一列的数组，多少行自己算
inputs = sc.transform(inputs)  # 使用MinMaxScaler将数据归一化
X_test = []
for i in range(60, 76):
    X_test.append(inputs[i-60:i, 0])   # 重新规整测试数据集
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  #  将X_test的数据转化到3D维度的数组中，时间步长与训练集一样，每一步表示一个特征。
predicted_stock_price = regressor.predict(X_test)   # 回归预测
predicted_stock_price = sc.inverse_transform(predicted_stock_price)   # 将预测的股价逆变换正常股价

# Visualising the results  可视化结果
plt.plot(real_stock_price, color = 'red', label = 'Real TATA Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted TAT Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.show()