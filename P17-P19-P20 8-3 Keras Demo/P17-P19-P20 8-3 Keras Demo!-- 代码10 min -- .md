[TOC]



# P17 8-3: Keras Demo <!-- code 10' --> 

x_train是一个二维的向量，x_train.shape=(10000,784)
train data一共有10000笔，每笔由一个784维的vector所表示。数值就是灰度，越大越偏向黑色
⦁	y_train也是一个二维向量，y_train.shape=(10000,10)，其中只有一维的数字是1，其余的为0。

```python
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.datasets import mnist
```



```python
def load_data():
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    number=10000
    x_train=x_train[0:number]
    y_train=y_train[0:number]
    x_train=x_train.reshape(number,28*28)
    x_test=x_test.reshape(x_test.shape[0],28*28)
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    # convert class vectors to binary class matrices
    y_train=np_utils.to_categorical(y_train,10)
    y_test=np_utils.to_categorical(y_test,10)
    x_train=x_train
    x_test=x_test
    # x_test=np.random.normal(x_test)
    x_train=x_train/255
    x_test=x_test/255
    return (x_train,y_train),(x_test,y_test)
```



```python
if __name__ == '__main__':
    # load training data and testing data
    (x_train, y_train), (x_test, y_test) = load_data()
    # x_train 是一个2维的向量，第一个维度是 10000，第二个维度dimension 是784，告诉我们 training data 总共有10000笔，每一笔由784维的vector表示的。在这个vector里面多数的值是0，少部分值介于0~1之间，代表这个 pixel 有没有被涂黑，涂得最黑就是1，所以这个数值代表pixel的颜色有多深
	print(x_train.shape) #(10000,784)
    # y_train 是label，第一个dimension是10000维，第二个维度 dimension 是10维
    print(y_train.shape) #(10000,10)
    #第一笔date label拿出来看看，多数数值是0，只有某一个维度数字是1，从0开始算，第一个dimension对应0，对应5的维度是1，意味着x_train[0]中数字代表数字5
    print(y_train[0])#array([0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,])
    
    # define network structure ,units=689 ,P19 activation='relu'
    model = Sequential()

    model.add(Dense(input_dim=28*28, units=633, activation='sigmoid'))
    model.add(Dropout(0.7)) # P19
    model.add(Dense(units=633, activation='sigmoid')) 
    model.add(Dropout(0.7))  # P19
    model.add(Dense(units=633, activation='sigmoid')) 
    model.add(Dropout(0.7)) # P19
    #for i in range(10): # 第三次加上循环并不work ,
	#	model.add(Dense(units=689,activation='sigmoid'))
    model.add(Dense(units=10, activation='softmax'))
    

    # set configurations #P19 categorical_crossentropy ，adam ，P17 mse，SGD
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.1),
                  metrics=['accuracy'])

    # train model P17 batch_size=100 ，P19 batch_size=10000 batch_size=1
    model.fit(x_train, y_train, batch_size=100, epochs=20)#P19 

    # evaluate the model and output the accuracy
    result_train = model.evaluate(x_train, y_train) #P19
    print('Train Acc:', result_train[1]) #P19
    
    result = model.evaluate(x_test, y_test)
    print('Test Acc:', result[1])
```

第一次：正确率 11.35%

第二次：units 由633 改成 689 ，正确率 21.66%

第三次：加上10层循环正确率 还是 11.35%

*deep learning 并不是越deep越好，隐层Neure调整，对整体效果也不一定有助益*
*关于deep learning 的实践，还是需要基于理论基础，而不是参数随便调来调去，所以继续跟着课程好好学。*



# P19 9-2: Keras Demo2 <!-- code 15' -->

deep learning在training 的时候非常容易train 坏掉。需要看一下 training set,看一下有没有把它的能力做起来，如果在training set 上overfitting 都做不到，在testing date上也不会有好结果

## 看一下 training set 的结果

```python
    result_train = model.evaluate(x_train, y_train) #P19
    print('Train Acc:', result_train[1]) #P19 Train Acc:11.26
```

training set 的 accuracy 也是差的，network在train 的时候就没train 好，要想办法在 training set 得到好的 Performance

## loss function 设的不对

分类问题mse不适合，将loss 中 mse 改为categorical_crossentropy，Training set accuracy =87.34%,Testing set accuracy =85.82%

```python
model.compile(loss='categorical_crossentropy',#P19 categorical_crossentropy P17 mse
              optimizer=SGD(lr=0.1),#adam
              metrics=['accuracy'])
```
## batch_size 对结果造成的影响

把batch_size由100 改到10000,跑超快(GPU 平行运算)，一样的network架构，batch_size太大，performer就坏掉，Training set accuracy =11.26%,Testing set accuracy =11.35%

```python
model.fit(x_train, y_train, batch_size=10000, epochs=20)
```
把batch_size 改到1，GPU没有办法发挥它的平行运算的效能，所以跑得超慢~

```python
model.fit(x_train, y_train, batch_size=1, epochs=20)
```

## activation function

layer用10层,看Training set accuracy 没有train 起来，卡住了

```python
for i in range(10): 
	model.add(Dense(units=689,activation='sigmoid'))
```
改一下activation function，把sigmoid都改为**relu**，Training accuracy将近100%，tTesting accuracy =95.64%

    model.add(Dense(input_dim=28*28, units=689, activation='relu'))
    model.add(Dense(units=689, activation='relu')) 
    model.add(Dense(units=689, activation='relu'))
    for i in range(10): 
    	model.add(Dense(units=689, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
## normalize

现在我们的image有normalize的，每个pixel我们用一个0-1之间的值进行表示，1代表最黑，0代表没有涂黑。image通常我们是用 灰阶来表示它，每一个pixel的值用0~255来表示，所以用x_train/255 来做normalize。注释掉x_train=x_train/255，又做不起来了

**这种小小的有没有做 normalize 的地方对结果有关键的影响**

```python
# x_train=x_train/255
```

## optimizer 

把SGD改为Adam，用adam的时候最后收敛的地方差不多，但是上升的速度变快,用 adam第一个epoch正确率就有85%

```python
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```
## Dropout

在test set上每个 pixel随机加上noise，train set 没有noise，结果就烂掉了

```python
 x_test=np.random.normal(x_test)
```

加在每个hidden layer后面，要知道dropout加入之后，train的效果会变差（overfitting才加drop out），然而test的正确率提升了,Training set accuracy =99%,Testing set accuracy =61%(加了noise)

```python
model.add(Dense(input_dim=28*28,units=689,activation='relu'))
model.add(Dropout(0.7))# 通常 Dropout(0.5)
model.add(Dense(units=689,activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(units=689,activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(units=10,activation='softmax'))
```



# P20 9-3: Fizz Buzz in Tensorflow (sequel)  <!-- 6' -->

## 硬 ”train“ 一发的代表故事

看起来好像不能train 的东西，但是我们还是用train的方法来解决，叫做硬”train“一发

## fizz buzz

部落格(台湾博客称呼) 里面，有一个 fizz buzz in Tensorflow 的故事。一个人面试写  fizz buzz 程序。 fizz buzz 就是现在有一串数字，比如1~100，如果这串数字里面这个数字可以被3整除就output fizz ，可以被5整除，就output buzz，同时可以被3和5整除，就output  fizz buzz 。比如1~16输出为 1，2，fizz,4,buzz,fizz,7,8,fizz,buzz,11,fizz,13,14,fizz buzz,16....

面试者要 import 一些 package，准备一些train data，先要label 101到1000的 fizz buzz（Amazon），我们做一个network ，给他train 下去，看看结果

## 实作

对数字101到1000做了labeling，先来看一下 training data，每一笔就代表了一个数字，一共900笔data.

把第一笔data打印出来,每一个数字都是用二进位来表示。第一个数字是101，用二进位来表示即为[1,0,1,0,0,1,1,0,0,0]，每一位表示2^0^ ，其他位依次表示2^{n-1}，n表示左数第几位。

label 的data，比如101不可以被3或者5整除，他output原来自己的数值，y有四个class，分别代表output 原来的数字，output fizz，output buzz,output fizzbuzz.101 output原来的数字就是  [1,0,0,0]，101 output fizz 就是  [0,1,0,0]

```python
print(x_train.shape)
>> (900,10)
print(x_train[0])# 把第一笔data打印出来
>> [1,0,1,0,0,1,1,0,0,0]
print(y_train[0])# label 的data
>> [1,0,0,0]
```

```python
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD,Adam
import numpy as np

def fizzbuzz(start,end):
    x_train,y_train=[],[]
    for i in range(start,end+1):
        num = i
        tmp=[0]*10
        j=0
        while num :
            tmp[j] = num & 1
            num = num>>1
            j+=1        
        x_train.append(tmp)
        if i % 3 == 0 and i % 5 ==0:
            y_train.append([0,0,0,1])
        elif i % 3 == 0:
            y_train.append([0,1,0,0])
        elif i % 5 == 0:
            y_train.append([0,0,1,0])
        else :
            y_train.append([1,0,0,0])
    return np.array(x_train),np.array(y_train)

x_train,y_train = fizzbuzz(101,1000) #打标记函数
x_test,y_test = fizzbuzz(1,100)
# network 架构 
model = Sequential()
model.add(Dense(input_dim=10,output_dim=100))
model.add(Activation('relu'))
model.add(Dense(output_dim=4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=20,nb_epoch=100)

result = model.evaluate(x_test,y_test,batch_size=1000)
	   print('Acc：',result[1])
```

正确率是76%

结果并没有达到百分百正确率，然而并不会放弃，所以我们首先开一个更大的neure，把hidden 从100改到1000

```python
model.add(Dense(input_dim=10,output_dim=1000))
```


再跑一跑，跑起来了，跑到100了，正确率就是100%


、