
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense_1 (Dense)              (None, 64)                640064    
# _________________________________________________________________
# dense_2 (Dense)              (None, 64)                4160      
# _________________________________________________________________
# dense_3 (Dense)              (None, 46)                2990      
# =================================================================
# Total params: 647,214
# Trainable params: 647,214
# Non-trainable params: 0
# _________________________________________________________________


#数据处理
from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]


#搭建模型
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#开始训练
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

#预测
predictions = model.predict(x_test)

#另一种标签处理方式
y_train = np.array(train_labels)
y_test = np.array(test_labels)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
