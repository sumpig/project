# 1 - 原始数据集
>链接：https://pan.baidu.com/s/1Mc1dbsHGQ0SOUlZFyRynIw 
提取码：70k8 

# 2 - 创建用于训练和测试的数据集
>分为猫狗两个类别，其中每个类别中，训练集为1000个样本，验证集为500个样本，测试集为500个样本

```python
import os, shutil

original_dataset_dir = 'kaggle_original_data'

base_dir = 'cats_and_dogs_small'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

#训练集
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

#验证集
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

#测试集
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
```

# 3 - 构建网络
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
```
