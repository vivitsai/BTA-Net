from __future__ import print_function

import csv
from idlelib.idle_test.test_browser import C1, C2

from keras.utils import np_utils
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Flatten, Conv1D, MaxPooling1D, Dense, Activation, \
    Dropout, GlobalMaxPooling1D, AveragePooling2D, ConvLSTM2D, GlobalMaxPooling2D, Recurrent, Reshape, Bidirectional, \
    BatchNormalization, Merge, concatenate, activations, merge, add, Multiply
from keras.utils import  to_categorical
from keras.layers import Embedding, Permute, AveragePooling1D
from pandas.core.frame import DataFrame
from keras.engine import Layer
from keras.layers import MaxPooling1D
from keras.optimizers import Adam
import pandas as pd
import sklearn.model_selection
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from keras import regularizers, Model, Input
import random
import seaborn as sns


from sklearn.model_selection import train_test_split

def attention_spatial(inputs2):

    a = Dense((inputs2.shape[3]).value, activation='softmax')(inputs2)
    # a = Permute((1, 3, 2))(a)
    # a = Reshape(((inputs2.shape[3] * 2).value, (inputs2.shape[1] * inputs2.shape[2]).value))(a)
    # a = MaxPooling1D(4)(a)
    # a = Conv1D(filters=16, kernel_size=7, strides=4)(a)
    # a = Conv1D(filters=16, kernel_size=7, activation='relu', strides=2)(a)
    # a = Reshape(((a.shape[2] //4).value, (a.shape[2] //4).value, (a.shape[1] ).value))(a)
    return a





def attention_vertical(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim1 = int(inputs.shape[1])
    input_dim2 = int(inputs.shape[2])
    input_dim3 = int(inputs.shape[3])

    a = Permute((3, 1,2))(inputs)
    a = Reshape((input_dim3, input_dim2,input_dim1))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim2, activation='softmax')(a)


    a_probs = Permute((3,2,1))(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return a_probs


def attention_horizontal(inputs2):
    input_dim1 = int(inputs2.shape[1])
    input_dim2 = int(inputs2.shape[2])
    input_dim3 = int(inputs2.shape[3])

    a = Permute((3, 2,1))(inputs2)
    a = Reshape((input_dim3, input_dim2,input_dim1 ))(a) # this line is not useful. It's just to know which dimension is what.

    a = Dense(input_dim2, activation='softmax')(a)

    b_probs = Permute((3,2,1))(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return b_probs

class TDA(Layer):

    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = activations.get(activation)
        super(TDA, self).__init__(**kwargs)


    def call(self, x):
        assert isinstance(x, list)

        X, A1,A2,A3 = x
        A = (A1 + A2 + A3)
        concatenate2 = tf.multiply(A, X)


        max1 = tf.maximum(A1, A2)

        max = tf.maximum(max1, A3)
        concatenate3 = K.concatenate([X, max], axis=3)

        # CAG = tf.multiply(A1, A2)
        # ADD = CAG + A3
        # concatenate2 = ADD + X

        # AA1 = (A1 + A2 + A3) / 3
        # AA2 = (tf.multiply((AA1 - A2), (AA1 - A2)) + tf.multiply((AA1 - A3), (AA1 - A3)) + tf.multiply((AA1 - A1),(AA1 - A1)))
        # AA2 = K.sqrt(AA2 / 2)
        # concatenate3 = K.concatenate([X, ADD], axis=3)
        return [concatenate2 , concatenate3]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        image_size = input_shape[0][1]
        input_dim1 = (input_shape[0][0],image_size,image_size, 1 * self.units)
        input_dim2 = (input_shape[0][0], image_size, image_size, 2 * self.units)
        return [input_dim1, input_dim2]


class BC(Layer):

    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = activations.get(activation)
        super(BC, self).__init__(**kwargs)


    def call(self, x):
        assert isinstance(x, list)

        M1, M2 = x
        reward = 2*M1
        punishment = tf.zeros_like(M1)
        M1 = tf.where(M1 > 0.2, x=reward, y=punishment)

        A = tf.multiply(M1, M2)

        return A

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        image_size = input_shape[0][1]
        input_dim = (input_shape[0][0],image_size,image_size, 1 * self.units)
        return input_dim



new_path1 = r''  #input
new_path2 = r''     #label

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,  decay=3e-8)


subtrainfeature = pd.read_csv(r'C:\Users\Administrator\Desktop\3\indian\indian_feature_1.csv')  #feature
subtrainLabel = pd.read_csv(r'C:\Users\Administrator\Desktop\3\indian\indian_label_1_101.csv')   #label

subtrain = pd.merge(subtrainLabel,subtrainfeature,on='Id')
from sklearn.utils import shuffle
# subtrain = shuffle(subtrain)
labels = subtrain.Class
subtrain.drop(["Class","Id"], axis=1, inplace=True)
subtrain = subtrain.values
(x_train, x_test,y_train,y_test)=train_test_split(subtrain,labels,test_size=0.99, stratify=labels)
# (x_train, x_test,y_train,y_test)=cross_validation.train_test_split(subtrain,labels,test_size=0.5)

print(y_train)
new_path3 = r'C:\Users\Administrator\Desktop\3\indian\indian_train2.csv'
# f = open(new_path3,'w')
# csv_write = f.writer(y_train,dialect='excel')
y_train.to_csv(new_path3,index=False,header=False)


# x_train = subtrain[0:10000]
x_test = subtrain[0:]
print(x_test.shape)
# # y_train = labels[0:10000]
y_test = labels[0:]

# x_train = subtrain[0:10000]
# x_test = subtrain[0:]
# # y_train = labels[0:10000]
# y_test = labels[0:]


# y_train = keras.utils.to_categorical(y_train, num_classes=2)
# y_test = keras.utils.to_categorical(y_test, num_classes=2)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

y_train=np_utils.to_categorical(y_train, num_classes=16)
y_test=np_utils.to_categorical(y_test, num_classes=16)



model1_7 = Input( shape=(100,))
x = Dense(units=256)(model1_7)
x = Reshape((16, 16,1))(x)

x1 = Conv2D(filters=64, kernel_size=11, activation='relu',strides=2, padding='same')(x)
x1 = BatchNormalization()(x1)

x2 = Conv2D(filters=64, kernel_size=7, activation='relu',strides=2, padding='same')(x)
x2 = BatchNormalization()(x2)

x3 = Conv2D(filters=64, kernel_size=3, activation='relu',strides=2, padding='same')(x)
x3 = BatchNormalization()(x3)

x = concatenate([x1,x2,x3])
# x = Conv2D(filters=96, kernel_size=1, activation='relu',strides=1, padding='same')(x)
x = AveragePooling2D(2,strides=1)(x)
x1 = BatchNormalization()(x)


xx = Conv2D(filters=128, kernel_size=3, activation='relu',strides=1, padding='same')(x)
m1 = Activation('sigmoid')(xx)
m2 = Activation('elu')(xx)
BC = BC(128)([m1, m2])

x = BatchNormalization()(BC)
att_3 = attention_spatial(x)
att_x2 = attention_vertical(x)
att_x = attention_horizontal(x)
G1,L2 = TDA(128)([x, att_x,att_x2,att_3])
L2 = Reshape((49,256))(L2)
L2 = Conv1D(filters=49, kernel_size=5,strides=2,activation='relu')(L2)
x = BatchNormalization()(x)
L2 = Reshape((7,7,23))(L2)
# L2 = BatchNormalization()(L2)
x = concatenate([G1,L2])
# x = Activation('relu')(x)
x = AveragePooling2D(2,strides=1)(x)
x2 = BatchNormalization()(x)


x = Conv2D(filters=128, kernel_size=3, activation='relu',strides=1, padding='same')(x2)
x = BatchNormalization()(x)
att_3 = attention_spatial(x)
att_x2 = attention_vertical(x)
att_x = attention_horizontal(x)
G1,L2 = TDA(128)([x, att_x,att_x2,att_3])
L2 = Reshape((36,256))(L2)
L2 = Conv1D(filters=36, kernel_size=5,strides=2,activation='relu')(L2)
x = BatchNormalization()(x)
L2 = Reshape((6,6,16))(L2)
# L2 = BatchNormalization()(L2)
x = concatenate([G1,L2])
x = AveragePooling2D(2,strides=1)(x)
x3 = BatchNormalization()(x)



x1 = AveragePooling2D(2)(x1)
x2 = AveragePooling2D(2)(x2)
x3 = AveragePooling2D(3,strides=1)(x3)


x = concatenate([x1,x2,x3])
x = Conv2D(filters=64, kernel_size=3, activation='relu',strides=1)(x)
x = BatchNormalization()(x)


x = Flatten()(x)# #

x = Dense(256,activation='relu')(x)
x = Dropout(0.5)(x)

all1_output = Dense(16)(x)
all1_output = Activation('softmax')(all1_output)


model1 = Model(inputs=[model1_7], outputs=[all1_output])
model1.summary()


model1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# model1.fit(x=x_train,y=y_train,batch_size=500,nb_epoch=200,verbose=2,validation_data=(x_test,y_test))

# loss,acc = model1.evaluate(x_test,y_test,verbose=2)
# print(acc)



from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath="best_model.h5",
                             monitor='val_acc',
                             verbose=1,
                             save_best_only='True',
                             save_weights_only='True',
                             mode='max',
                        period=1)


lrreduce=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.1, cooldown=0, min_lr=0)

import time
fit_start = time.clock()
history= model1.fit(x=x_train,y=y_train,batch_size=10000,epochs=400,verbose=2,validation_data=(x_test,y_test),callbacks = [checkpoint])
fit_end = time.clock()

print("train time is: ",fit_end-fit_start)

model1.load_weights('best_model.h5')

t_start = time.clock()
loss,acc = model1.evaluate(x_test,y_test,verbose=2)
t_end = time.clock()
print('Test loss :',loss)
print('Test accuracy :',acc)
print("test time is: ",t_end-t_start)





y_pred_class = model1.predict(x_test)

data1 = DataFrame(y_pred_class)
data1.to_csv(r'C:\Users\Administrator\Desktop\3\indian\indian_结果标签.csv',index=False,header=False)


'----------------------------------------------------------------------------------------------------------------------'
list3 = []


lines = y_pred_class.tolist()



f=open(new_path1, mode='w')

a = 0
for line in lines:
    if line:
        a = a + 1
        # print(line.index(max(line)))
        f.write(str(line.index(max(line))))
        f.write('\n')
        list3.append(line.index(max(line)))
        # list.append('\n')
f.close()

list22 =[ ]
f2 = open(new_path2,mode='r')
list_true = f2.readlines()

for i in list_true:
    ii = i
    ii = ii.replace('\n','')
    ii = int(ii)
    list22.append(ii)

# print(list)

"----------------------------------------------------------------------------------------------------------------------"
all_path = r'C:\Users\Administrator\Desktop\3\indian'

path1 = all_path + '\indian_label.csv'
path2 = all_path + '\indian_结果标签_1.csv'
path3 = all_path + '\indian_label_zui.csv'



f1=open(path1, mode='r')
f2=open(path2, mode='r')
f3=open(path3, mode='w')

line1 = f1.readlines()
line2 = f2.readlines()

a = 0

for i1 in line1:
    if i1 != '16\n':
        i1 = line2[a]
        a = a + 1
        # print(a)
        f3.write(i1)
    else:
        f3.write(i1)
        continue

f3.close()
"----------------------------------------------------------------------------------------------------------------------"
import numpy as np
aa = 0
path3 = all_path + '\indian_label_zui.csv'
f3=open(path3, mode='r')
list = f3.readlines()
for i in range(len(list)):
    aa = aa + 1
    if list[i] == '1\n':
        list[i] =[255,255,102]
    elif list[i] == '2\n':
        list[i] =[0,48,205]
    elif list[i] == '3\n':
        list[i] = [255, 102, 0]
    elif list[i] == '4\n':
        list[i] =[0,255,154]
    elif list[i] == '5\n':
        list[i] =[255,48,205]
    elif list[i] == '6\n':
        list[i] =[102,0,255]
    elif list[i] == '7\n':
        list[i] =[0,154,255]
    elif list[i] == '8\n':
        list[i] =[0,255,0]
    elif list[i] == '9\n':
        list[i] =[129,129,0]
    elif list[i] == '10\n':
        list[i] = [129, 0, 129]
    elif list[i] == '11\n':
        list[i] = [48, 205, 205]
    elif list[i] == '12\n':
        list[i] = [0, 102, 102]
    elif list[i] == '13\n':
        list[i] = [48, 205, 48]
    elif list[i] == '14\n':
        list[i] = [102, 48, 0]
    elif list[i] == '15\n':
        list[i] = [102, 255, 255]
    elif list[i] == '0\n':
        list[i] = [255, 255, 0]
    else:
        list[i] = [0, 0, 0]
data = np.reshape(list, (145, 145, 3))
import scipy.misc
scipy.misc.imsave(r'C:\Users\Administrator\Desktop\3\indian\CNN_indian_0.01_BCTDA.jpg', data)

print('11')





from sklearn import metrics


def kappa(confusion_matrix, k):
    dataMat = np.mat(confusion_matrix)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe  = float(ysum*xsum)/np.sum(dataMat)**2
    OA = float(P0/np.sum(dataMat)*1.0)
    cohens_coefficient = float((OA-Pe)/(1-Pe))
    return cohens_coefficient

classify_report = metrics.classification_report(list22, list3)
confusion_matrix = metrics.confusion_matrix(list22, list3)
overall_accuracy = metrics.accuracy_score(list22, list3)
acc_for_each_class = metrics.precision_score(list22, list3, average=None)
average_accuracy = np.mean(acc_for_each_class)
kappa_coefficient = kappa(confusion_matrix, 16)


#
cm = np.array(confusion_matrix)
matrix_label =['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16']
plt.figure(figsize=(12,10), dpi= 100)
sns.heatmap(cm, xticklabels=matrix_label, yticklabels=matrix_label, center=0, annot=True,fmt="d",linewidths=.5,cbar=False)

# Decorations
#plt.title('Indian Pines (15%)', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('confusion_matrix.png')
plt.show()

print('classify_report : \n', classify_report)
print('confusion_matrix : \n', confusion_matrix)
print('acc_for_each_class : \n', acc_for_each_class)
print('average_accuracy: {0:f}'.format(average_accuracy))
print('overall_accuracy: {0:f}'.format(overall_accuracy))
print('kappa coefficient: {0:f}'.format(kappa_coefficient))

newpath = r'C:\Users\Administrator\Desktop\3\indian\CNN_indian_0.05_BCTDA.txt'
f = open(newpath,'w')
f.write(classify_report)
#f.write(confusion_matrix)
f.write(str(acc_for_each_class.tolist()))
f.write('\n')
f.write('average_accuracy:{0:f}'.format(average_accuracy))
f.write('\n')
f.write('overall_accuracy:{0:f}'.format(overall_accuracy))
f.write('\n')
f.write('kappa coefficient:{0:f}'.format(kappa_coefficient))
f.write('\n')

f.close()
