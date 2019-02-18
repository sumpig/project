# 图片分类 - Dogs vs cats
深度学习想要独立地在训练数据中找到有趣的特征，需要拥有大量训练样本时才能实现。要想在小数据集中训练模型，就要用到卷积神经网络局部平移不变的特性。深度学习模型本质上具有高度的可复用性，特别是在计算机视觉领域，许多与训练模型可以用于在数据很少的情况下构建强大的视觉模型。这里使用 Kaggle 的 dogs-vs-cats 数据（https://www.kaggle.com/c/dogs-vs-cats/data) 中的一小部分，利用数据增强，并结合 VGG16 预训练模型来训练，从而达到一个较高的精度。

**point**
- 数据增强
- VGG16
- fine-tuning

## 1 - 提取数据
这个数据集包含25000张猫狗图片，每个类别都有12500张。我们需要创建一个新数据集，其中包含三个子集：每个类别各1000样本的训练集，500个样本的验证集合500个样本的测试集。

```python
# 原始数据集
original_dataset_dir = r"H:\kaggle\train"

# 保存较小数据集的目录
base_dir = r"H:\kaggle\cats_and_dogs_small"
os.mkdir(base_dir)

# 训练、验证和测试目录
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# 猫的训练图像目录
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

# 狗的训练图像目录
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# 猫的验证图像目录
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

# 狗的验证图像目录
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

# 猫的测试图像目录
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

# 狗的测试图像目录
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# 将前1000张猫的图像复制到train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# 将接下来500张猫的图像复制到validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# 将接下来500张猫的图像复制到test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# 将前1000张狗的图像复制到train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# 将接下来500张狗的图像复制到validation_dogs_dir    
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

# 将接下来500张狗的图像复制到test_s_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
```


## 2 - 构建网络
我们构建一个小型卷积神经网络，卷积神经网络由Conv2D层（使用relu激活）和 MaxPooling2D层交替堆叠构成。初始输入尺寸为150\*150。这是一个二分类问题，所以网络最后一层使用sigmoid激活的单一单元。这个单元将对某个类别的概率进行编码。

```python
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d_1 (Conv2D)            (None, 148, 148, 32)      896       
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 74, 74, 32)        0         
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 72, 72, 64)        18496     
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 36, 36, 64)        0         
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 34, 34, 128)       73856     
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 (None, 17, 17, 128)       0         
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 15, 15, 128)       147584    
# _________________________________________________________________
# max_pooling2d_4 (MaxPooling2 (None, 7, 7, 128)         0         
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 6272)              0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 512)               3211776   
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 513       
# =================================================================
# Total params: 3,453,121
# Trainable params: 3,453,121
# Non-trainable params: 0
# _________________________________________________________________


from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 配置模型用于训练
from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
```

## 3 - 数据预处理
数据是以JPEG文件的形式保存在硬盘中，应该将数据格式化为经过预处理的浮点数张量，步骤如下：
1. 读取图像文件。
2. 将JPEG文件解码为RGB像素网格。
3. 将这些像素网格转换为浮点数张量。
4. 将像素值（0~255）缩放到[0,1]区间。

Keras有一个图像处理辅助工具的模块，位于 keras.preprocessing.image , 它包含 [ImageDataGenerator](https://keras.io/zh/preprocessing/image/#imagedatagenerator) 类，可以快速创建Python生成器，能够将硬盘上的图像文件自动转换为预处理好的张量批量。

```python
from keras.preprocessing.image import ImageDataGenerator

# 将所有图像缩放
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # 目标目录
        train_dir,
        # 将所有图像大小调整为150*150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# 利用批量生成器拟合模型
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

# 保存模型
model.save('cats_and_dogs_small_1.h5')
```

- 绘制训练过程中的损失曲线和精度曲线
```python
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```

![alt](https://github.com/sumpig/project/blob/master/picture/%E7%8C%AB%E7%8B%971.png)

这些图像可以很明显的看出过拟合的特征。验证精度停留在70%~72%。验证损失在5轮后达到最小值，然后保持不变，而训练损失则一直下降，直到接近0。因为训练样本较少，所以过拟合问题很大。下面我们使用数据增强和dropout的方法来重新训练下。

# 5 - 使用数据增强
- 定义一个包含dropout的新卷积神经网络
```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
```

- 利用数据增强生成器训练卷积神经网络
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

#验证数据
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)
      
model.save('cats_and_dogs_small_2.h5')      
```

- 再次绘制结果
```python
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```

![alt](https://github.com/sumpig/project/blob/master/picture/%E7%8C%AB%E7%8B%972.png)

>使用正则化方法后，精度提升到86%。再想提升精度十分困难，因为可用的数据太少。

# 6 - 使用预训练的卷积神经网络
> 使用ImageNet上训练的VGG16网络的卷积基从猫狗图像中提取有趣的特征，然后再这些特征上训练一个猫狗分类器。

- 将VGG16卷积基实例化
```python
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
                 
conv_base.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         (None, 150, 150, 3)       0         
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________
```

- 不使用数据增强的快速特征提取
```python
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = 'cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

#数据预处理
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

#特征提取
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        #提取特征
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

#定义并训练密集连接分类器
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))
                    
#绘制结果
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```

![alt](https://github.com/sumpig/project/blob/master/picture/%E7%8C%AB%E7%8B%973.png)

>精度达到了90%，从图中可以看出，虽然dropout比率相当大，但模型几乎从一开始就过拟合，这是因为本方法没有使用数据增强。

- 使用数据增强的特征提取
```python
from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# vgg16 (Model)                (None, 4, 4, 512)         14714688  
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 8192)              0         
# _________________________________________________________________
# dense_3 (Dense)              (None, 256)               2097408   
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 257       
# =================================================================
# Total params: 16,812,353
# Trainable params: 16,812,353
# Non-trainable params: 0
# _________________________________________________________________

#冻结VGG16的卷积基
conv_base.trainable = False

#训练模型
```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

#测试集
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)
```
![alt](https://github.com/sumpig/project/blob/master/picture/%E7%8C%AB%E7%8B%974.png)

>精度为96%

# 7 - 微调模型
>对于用于特征提取的冻结的模型基，微调是指将其顶部的几层“解冻”，并将这解冻的几层和新增加的部分联合训练。目的是略微调整所复用模型中更加抽象的表示，以便让这些表示与当前的问题更加相关。

- 微调VGG16中的最后三个卷积层
```python
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
```

- 训练模型
```python
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)
      
model.save('cats_and_dogs_small_4.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```
![alt](https://github.com/sumpig/project/blob/master/picture/%E7%8C%AB%E7%8B%975.png)

- 使绘制曲线变得平滑
```python
def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

plt.plot(epochs,
         smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,
         smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,
         smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,
         smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```
![alt](https://github.com/sumpig/project/blob/master/picture/%E7%8C%AB%E7%8B%976.png)

>精度提高了1%

- 在测试数据上最终评估这个模型
```python
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
```
>最终，仅用一小部分数据得到了97%的精度

# 8 - 小结
- 在小型数据集上主要的问题是过拟合，数据增强是一种降低过拟合的强大方法。
- 利用特征提取，可以很容易将现有的卷积神经网络复用于新的数据集。
- 作为特征提取的补充，微调可以进一步提高模型性能。
