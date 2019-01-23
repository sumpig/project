- 一层模型

```python
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#dense_5 (Dense)              (None, 512)               401920    
#_________________________________________________________________
#dense_6 (Dense)              (None, 10)                5130      
#=================================================================
#Total params: 407,050
#Trainable params: 407,050
#Non-trainable params: 0
#_________________________________________________________________


import keras

#获取数据
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#数据处理
train_images = train_images.reshape((60000, -1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, -1))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels) #将整数转换为向量：5 变换为 array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], dtype=float32)
test_labels = to_categorical(test_labels)

#搭建模型
from keras import models, layers

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

#开始训练
model.fit(train_images, train_labels, epochs=5, batch_size=128)

#测试集精度预测
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
```

- 三层卷积模型
```python
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0         
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 11, 11, 64)        18496     
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0         
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 3, 3, 64)          36928     
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 576)               0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 64)                36928     
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                650       
# =================================================================
# Total params: 93,322
# Trainable params: 93,322
# Non-trainable params: 0
# _________________________________________________________________


import keras

#数据处理
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#搭建模型
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
#开始训练
model.fit(train_images, train_labels, epochs=5, batch_size=64)

#测试集预测
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
```
