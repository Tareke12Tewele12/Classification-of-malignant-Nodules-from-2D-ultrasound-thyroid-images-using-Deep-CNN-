#!/usr/bin/env python
# coding: utf-8

###Thyroid nodule Classification

import warnings

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# In[2]:
## Importing library 
seed_value= 0
### importing libraries
import tensorflow as tf
tf.random.set_seed(seed_value)
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random as python_random
python_random.seed(123)
import random
import numpy as np
#np.random.seed(1142)
import cv2
import math
###tf.random.set_seed(1234)
import keras
import keras.backend as K
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, add, GlobalAveragePooling2D
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score,roc_curve, confusion_matrix, roc_auc_score, auc, f1_score
from keras.regularizers import l2
import tensorflow as tf
from keras import layers
from keras import Model
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
plt.rcParams.update({'font.size': 20})
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ReduceLROnPlateau, Callback, CSVLogger,EarlyStopping, ModelCheckpoint
from keras.applications.densenet import DenseNet121
from keras.models import load_model
from keras.layers import LeakyReLU
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from IPython.display import Image, display
import matplotlib.image as mpimg
import seaborn
from sklearn.metrics import confusion_matrix
import itertools

                                                      #from utils.visualize import plotImages, plotHistory, explainGradCam
# In[6]:
import keras
import tensorflow as tf
import keras.backend as K
print("Keras Version", keras.__version__)
print("tensorflow Version", tf.__version__)
print("dim_ordering:", K.image_data_format())

train_dir = '../Data_train/transverse/'               #Dataset for Sagittal 
test_dir = '../Data_train/val/'

# In[4]:
#train_dir = '../Data_train/longitudinal/'               #Dataset for longitudinal 
#test_dir = '../Data_train/val/'

#train_dir = '../Dataset_final_open/train/'               #Dataset final removed artifact 
#test_dir = '../Dataset_final_open/val/'

#train_dir = '../Dataset_final_open/train/'               #Dataset final with artifact
#test_dir = '../Dataset_final_open/val/'

### initial parameters
output_classes = 2
batch_size =8
img_height, img_width = 800, 600
input_shape = (img_height, img_width, 3)
epochs = 150
batch_val =1
# In[7]:ss

## Checking reading files from the directory
for root,dirs,files in os.walk(train_dir):
    print (root, len(files))
print("*"*30)
for root,dirs,files in os.walk(test_dir):
    print (root, len(files))

# In[8]:
### Assignment of weights 
classes_dict = {
    'benign': 0,
    'malign': 1,

}
'''
weight_for_0 =  460 / (280 * 2.0)   #automatic class_weight assignment! 
weight_for_1 =  460/ (180 *2.0)
class_weights= {0: weight_for_0, 1: weight_for_1}

'''
class_weights={
    0: 1.5, # benign
    1: 2.5, # malign
    
}
# In[10]:
#Best_model ='DeepLearning/CheckPoints/Best_model.h5'
model_name="model_DenseNet"
#model_path = "DeepLearning/Models"
#loss = "categorical_crossentropy"
loss=SigmoidFocalCrossEntropy(alpha = 0.25,gamma= 3.50 ## focal loss 
metrics=["accuracy"]  # RMSE, ROC, AUC, Kappa, ..etc.

# In[11]:
import time
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# In[12]:
checkpoint_path = "DeepLearning/DenseNet121/"
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, mode ="auto", factor=0.8, min_lr=0.00001)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    #filepath_1=checkpoint_path1,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1,
    save_best_only=True)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=15,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)

time_callback = TimeHistory()
logger_path = "DeepLearning/"+model_name+".log"
csv_logger = CSVLogger(logger_path, separator=',', append=False)
my_callbacks = [reduce_lr, model_checkpoint_callback, early_stopping, time_callback]

# In[13]:
                                                                               ### Adding varaiblity/preprocessing 
def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img
                                                                                  ### Weight  initialisation
tf.compat.v1.initializers.he_normal(                               
    seed=42
)
random_seed = np.random.seed(1142)

train_datagen = ImageDataGenerator(
                                    rescale=1./255, 
                                    featurewise_center=True, 
                                    samplewise_center=True, 
                                    featurewise_std_normalization=False, 
                                    samplewise_std_normalization=False, 
                                    zca_whitening=True, 
                                    zca_epsilon=1e-06,
                                    rotation_range=20, 
                                    width_shift_range=0.2, 
                                    height_shift_range=0.2, 
                                    brightness_range=[0.2,0.8],
                                    shear_range=0.1, 
                                    zoom_range=0.1, 
                                    channel_shift_range=0.1, 
                                    fill_mode='nearest', 
                                    cval=0.0, 
                                    horizontal_flip=True, 
                                    vertical_flip=True, 
                                    preprocessing_function=add_noise, 
                                    data_format='channels_last', 
                                    dtype='float32'
                                    )
'''

train_datagen = ImageDataGenerator(
                                    preprocessing_function=add_noise
                           )
'''

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    seed = random_seed,
    shuffle = True,
    subset = 'training',
    class_mode='categorical',
    classes = {'benign': 0,'malign': 1})

val_datagen= ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=60,
    width_shift_range=0.2,
    height_shift_range=0.2,
    featurewise_center=True,
    zoom_range= 0.2,
    shear_range=0.2,
    validation_split=0.15,        #15% of the training dataset are used to validate the model.
    horizontal_flip=True
)

validation_generator =val_datagen.flow_from_directory(
    train_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_val,
    class_mode='categorical',
    shuffle=False,
    subset='validation') # set as validation data

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    #featurewise_center=True,
    #rotation_range=60,
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width), 
    shuffle = False,
    class_mode='categorical',
    batch_size=batch_val,
)

# In[14]:
save_here ='../generated_images/'

for i in (train_datagen.flow_from_directory(train_dir,                    #image we chose
        save_to_dir=save_here,     #this is where we figure out where to save
         save_prefix='aug',        # it will save the images as 'aug_0912' some number for every new augmented image
        save_format='jpg'),range(50)) :     # here we define a range because we want 10 augmented images otherwise it will keep looping forever I think
        break

nb_train_samples = len(train_generator.filenames)
nb_test_samples = len(test_generator.filenames)
predict_size_train = int(math.ceil(nb_train_samples / batch_size))
predict_size_test = int(math.ceil(nb_test_samples / batch_val))

nb_test_samples = len(test_generator.filenames)
predict_size_test = int(math.ceil(nb_test_samples / batch_size))
num_class =len(train_generator.class_indices)

num_classes = len(test_generator.class_indices)
print("nb_train_samples:", nb_train_samples)
print("\npredict_size_train:", predict_size_train)
print("nb_test_samples:", nb_test_samples)
print("predict_size_test:", predict_size_test)

print("\n num_classes for Train:", num_class, "\n Num of class for validation", num_classes)

##############Attention Module   Ref: github: @EscVM
#In[]:

class ChannelAttention(tf.keras.layers.Layer):
      def __init__(self, filters, ratio):
        super(ChannelAttention, self).__init__()
        self.filters = filters
        self.ratio = ratio

        def build(self, input_shape):
            self.shared_layer_one = tf.keras.layers.Dense(self.filters//self.ratio,
                             activation='relu', kernel_initializer='he_normal', 
                              use_bias=True, 
                              bias_initializer='zeros')
            self.shared_layer_two = tf.keras.layers.Dense(self.filters,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

        def call(self, inputs):
            # AvgPool
            avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
            

            avg_pool = self.shared_layer_one(avg_pool)
            avg_pool = self.shared_layer_two(avg_pool)

            # MaxPool
            max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)
            max_pool = tf.keras.layers.Reshape((1,1,filters))(max_pool)

            max_pool = self.shared_layer_one(max_pool)
            max_pool = self.shared_layer_two(max_pool)


            attention = tf.keras.layers.Add()([avg_pool,max_pool])
            attention = tf.keras.layers.Activation('sigmoid')(attention)
            
            return tf.keras.layers.Multiply()([inputs, attention])

# In[]     #### Spatial Attention module with lambda =7
class SpatialAttention(tf.keras.layers.Layer):
      def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        
        def build(self, input_shape):
            self.conv2d = tf.keras.layers.Conv2D(filters = 1,
                    kernel_size=self.kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)

        def call(self, inputs):
            
            # AvgPool
            avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(inputs)
            
            # MaxPool
            max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(inputs)

            attention = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])

            attention = self.conv2d(attention)


            return tf.keras.layers.multiply([inputs, attention]) 

#In[]

pre_trained_model = DenseNet121(input_shape=input_shape, include_top=False, weights="imagenet")
'''
for layer in pre_trained_model.layers:
    print(layer.name)
    if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
        layer.trainable = True
        K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
        K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
    else:
        layer.trainable = False
         '''   
for layer in pre_trained_model.layers:
    print(layer.name)  
for layer in pre_trained_model.layers[-50:]:
    layer.trainable = True
print(len(pre_trained_model.layers))

last_layer = pre_trained_model.get_layer('conv_block16_concat')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output
pre_trained_model.layers[425].name
# In[ ]:
x=layers.SeparableConv2D(32, (5, 5), input_shape =input_shape)(last_output)
x=layers.BatchNormalization(axis=-1,momentum=0.99,center=True)(x)
x=ChannelAttention(64, 8)(x)                                                          ## channel attention
x=SpatialAttention(7)(x)                                                              ## Spatial Attention
x=layers.SeparableConv2D(32, (5, 5), input_shape =input_shape)(x)
x=ChannelAttention(128, 8)(x) 
x=SpatialAttention(11)(x)

x=layers.SeparableConv2D(32, (5, 5), input_shape =input_shape)(x)
x=layers.BatchNormalization(axis=-1,momentum=0.99, center=True)(x)
x = layers.GlobalMaxPooling2D()(x)                                            #Flatten the output layer to 1 dimension
print(x.shape)
x = layers.Dense(512, activation='relu')(x)                                   # Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dropout(0.25)(x)                                                   # Add a dropout rate of 0.5
x = layers.Dense(2, activation='softmax')(x)                                  # classification 


model = Model(pre_trained_model.input, x)                                     # Configure and compile the model
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True) 
model.compile(loss=SigmoidFocalCrossEntropy(alpha = 0.25,gamma= 3.50),
              optimizer=optimizer,
              metrics=['accuracy'])
print(len(model.layers))

# In[ ]:

model.summary()                                                               ### Model's architecture summary 

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples// batch_val,
    class_weight=class_weights,
    callbacks=[time_callback, model_checkpoint_callback,csv_logger, reduce_lr],
    verbose=1)
model.save(" My model_DenseNet")

#In[]
loss_val, acc_val = model.evaluate(test_generator, verbose=1,steps=nb_test_samples // batch_val)
print("Validation: accuracy = %f  ;  loss_v = %f" % (acc_val, loss_val))
score = model.evaluate_generator(test_generator, steps=32, verbose =1)

print ('Validation Score: ', score[0])
print ('Validation Accuracy: ',score[1])

plt.rcParams.update({'font.size': 12})

plt.style.use('seaborn-white')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Acc', 'val_Acc'], loc='lower right') # test acc
fig1=plt.savefig(" DenseNet_curve.png")
plt.close(fig1)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Test Loss'], loc='upper right')
fig2 =plt.savefig(" DenseNet_loss.png")
plt.close(fig2)
#plt.figure()
N =epochs
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
fig3=plt.savefig("test.png")
plt.close(fig3)
#In[]:


filename = test_generator.filenames
truth = test_generator.classes
label = test_generator.class_indices
indexlabel = dict((value, key) for key, value in label.items())

predicts = model.predict_generator(test_generator, steps=test_generator.samples/test_generator.batch_size, verbose=2)
print(predicts)
predict_class = np.argmax(predicts, axis=1)
print(predict_class)
errors = np.where(predict_class != truth)[0]
print("No of errors = {}/{}".format(len(errors),test_generator.samples))

plt.rcParams["axes.grid"] = True
plt.rcParams.update({'font.size': 20})

cm = confusion_matrix(truth,predict_class)
labels = []
for k,v in indexlabel.items():
    labels.append(v)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion Matrix')

    print(cm)
#     fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    fig4=plt.savefig('DenseNet_Confusion_Matrix.png', bbox_inches='tight', dpi = 100) 
    plt.figure(figsize=(10,10))
    plt.close(fig4)

plot_confusion_matrix(cm, classes=labels, title='Confusion Matrix\n')

#In[]:

print("classification_report\n")

print(classification_report(truth, predict_class))
print("\n")
print("\n")

#In[]

total=sum(sum(cm))

# accuracy = (cm[0,0]+cm[1,1]) / total
# print ('Accuracy : ', accuracy)

sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])
print('Sensitivity : ', sensitivity )

Specificity = cm[1,1]/(cm[1,1]+cm[0,1])
print('Specificity : ', Specificity )

print("\n")
print("\n")
print("\n")
#In[]
 
 ##### Griedent weighted- Class activation maps









