#!/usr/bin/env python
# coding: utf-8

###Thyroid nodule Classification

import warnings

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


# In[2]:
## Importing library 
import os
import random
import numpy as np
np.random.seed(777)
import cv2
import math
import keras
import keras.backend as K
import h5py
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold, StratifiedKFold
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, add, GlobalAveragePooling2D
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score,roc_curve, confusion_matrix, roc_auc_score, auc, f1_score
from keras.regularizers import l2

from keras import layers
from keras import Model
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
plt.rcParams.update({'font.size': 20})
from tensorflow.keras.preprocessing import image
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback, CSVLogger,EarlyStopping, ModelCheckpoint
from keras.applications.densenet import DenseNet121
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from keras_self_attention import SeqSelfAttention


# In[4]:
train_dir = '../Dataset_final_Art_NO_B3/train/' 
train_public ='../public_data/'                     #Dataset final without bethesda score B3 in validation set
test_dir = '../Dataset_final_Art_NO_B3/val/'

#train_dir = '../Dataset_final_open/train/'               #Dataset final with openning pre-processing 
#test_dir = '../Dataset_final_open/val/'

train_dir = '../Dataset_final/train/'               #Dataset final but not openning 
test_dir = '../Dataset_final/val/'

import keras
import tensorflow as tf
import keras.backend as K
print("Keras Version", keras.__version__)
print("tensorflow Version", tf.__version__)
print("dim_ordering:", K.image_data_format())

# In[6]:
###1
output_classes = 2   
batch_size =4
img_height, img_width = 800, 600
input_shape = (img_height, img_width, 3)
epochs =5
batch_val =1
# In[7]:ss

##1
for root,dirs,files in os.walk(train_dir):
    print (root, len(files))
print("*"*30)
for root,dirs,files in os.walk(test_dir):
    print (root, len(files))

# In[8]:
for root,dirs,files in os.walk(train_dir):
    print (root, len(files))
print("*"*30)
for root,dirs,files in os.walk(test_dir):
    print (root, len(files))

# In[9]:

classes_dict = {
    'benign': 0,
    'malign': 1,

}

weight_for_0 =  514 / (280 * 2.0)   #automatic class_weight assignment! 
weight_for_1 =  514/ (234 *2.0)
class_weights= {0: weight_for_0, 1: weight_for_1}

'''
class_weights={
    0: 1.5, # benign
    1: 3.5, # malign
    
}
'''
# In[10]:

#Best_model ='DeepLearning/CheckPoints/Best_model.h5'
model_name="model_DenseNet_cross"
#model_path = "DeepLearning/Models"
loss = "categorical_crossentropy"
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
checkpoint_path = "DeepLearning/cross_validation/"
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

def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

random_seed = np.random.seed(1142)

classes = {'../Dataset_final/train/benign': 0,'../Dataset_final/train/malign': 1}

train_datagen = ImageDataGenerator(
                                    rescale=1./255, 
                                    featurewise_center=True, 
                                    samplewise_center=True, 
                                    featurewise_std_normalization=True, 
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
                                    validation_split=0.0, 
                                    dtype='float32'
                                    )

'''
train_datagen = ImageDataGenerator(
                                    rescale=1./255
                                    )
'''

train_generator = train_datagen.flow_from_directory(
    train_dir,
    classes = {'benign': 0,'malign': 1})
    
train_generator_public = train_datagen.flow_from_directory(
    train_public,
    classes = {'malign': 1})
    
validation_datagen = ImageDataGenerator(
    rescale=1. / 255,
    featurewise_center=True,
    rotation_range=60,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

validation_generator = validation_datagen.flow_from_directory(
    test_dir)


# In[14]:

nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)
predict_size_train = int(math.ceil(nb_train_samples / 4))
predict_size_validation = int(math.ceil(nb_validation_samples / 1))

#nb_test_samples = len(test_generator.filenames)
#predict_size_test = int(math.ceil(nb_test_samples / batch_size))
num_class =len(train_generator.class_indices)

num_classes = len(validation_generator.class_indices)

print("nb_train_samples:", nb_train_samples)
print("nb_validation_samples:", nb_validation_samples)
print("\npredict_size_train:", predict_size_train)
print("predict_size_validation:", predict_size_validation)
#print("nb_test_samples:", nb_test_samples)
#print("predict_size_test:", predict_size_test)

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

# In[ ]:

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
    layer.trainable = False   
print(len(pre_trained_model.layers))

last_layer = pre_trained_model.get_layer('conv5_block16_concat')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output
pre_trained_model.layers[425].name
# In[ ]:
x=layers.SeparableConv2D(32, (5, 5), input_shape =input_shape)(last_output)
x=layers.BatchNormalization(axis=-1,momentum=0.99,center=True)(x)
x=ChannelAttention(64, 8)(x)                                                          ## channel attention
x=SpatialAttention(7)(x)                                                              ## Spatial Attention
x=layers.SeparableConv2D(32, (5, 5), input_shape =input_shape)(x)
x=SpatialAttention(9)(x)

x=layers.SeparableConv2D(32, (5, 5), input_shape =input_shape)(x)
x=layers.BatchNormalization(axis=-1,momentum=0.99, center=True)(x)
x = layers.GlobalMaxPooling2D()(x)                                            #Flatten the output layer to 1 dimension
print(x.shape)
x = layers.Dense(512, activation='relu')(x)                                   # Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dropout(0.25)(x)                                                   # Add a dropout rate of 0.5
x = layers.Dense(2, activation='softmax')(x)   #Output

# Configure and compile the model
model = Model(pre_trained_model.input, x)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True) 
model.compile(loss=SigmoidFocalCrossEntropy(alpha = 0.25,gamma= 3.50),
              optimizer=optimizer,
              metrics=['accuracy'])
print(len(model.layers))

# In[ ]:
Best_model ='Best_model.h5'
keras_callbacks   = [

      ModelCheckpoint(filepath= Best_model, monitor='val_loss', save_best_only=True, mode='min')
]
model.summary()

# In[ ]:
kf = KFold(n_splits = 5)
                         
k_folds =5
## Training with K-fold cross validation
kf = KFold(n_splits=k_folds, shuffle=True, random_state=1)
kf.get_n_splits(train_dir)

X = np.array(train_generator)
y = np.array(train_generator.classes)
x1 =np.array(train_generator_public)
y1 =np.array(train_generator_public.classes)

i = 1
for train_index, test_index in kf.split(X):
    trainData = X[train_index]
    testData = X[test_index]
    trainData= trainData.astype('float32')
    testData = testData.astype('float32')
    trainLabels = y[train_index]
    testLabels = y[test_index]
    trainData =trainData.append(x1)
    trainLabels =trainLabels.append(y1)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        seed = random_seed,
        shuffle = True,
        subset = 'training',
        class_mode='categorical',
        classes = {'benign': 0,'malign': 1})
    
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=(testData , testLabels), 
        validation_steps=nb_validation_samples // batch_val,
        class_weight=class_weights,
        callbacks=([time_callback, model_checkpoint_callback,csv_logger, reduce_lr], keras_callbacks),
        verbose=2)
     
    validation_generator = validation_datagen.flow_from_directory(
            test_dir,
            target_size=(img_height, img_width),
            batch_size=batch_val,
            seed = random_seed,
            shuffle = False,
            class_mode='categorical')
    accuracy = model.evaluate(validation_generator, verbose=1,steps=nb_validation_samples // batch_val)
    print("=========================================")
    print("Average_accuray K_",i,"=:" , accuracy )
    print("=========================================")
    
    #Model =tf.keras.models.load_model('Best_model.h5')
    loss_val, acc_val = model.evaluate(validation_generator, verbose=1,steps=nb_validation_samples // batch_val)
    print("Validation: accuracy = %f  ;  loss_v = %f" % (acc_val, loss_val))
    score = model.evaluate_generator(validation_generator, steps=32, verbose =1)

    print ('Validation Score: ', score[0])
    print ('Validation Accuracy: ',score[1])
    import seaborn
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

    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix, classification_report
    filename = validation_generator.filenames
    truth = validation_generator.classes
    label = validation_generator.class_indices
    indexlabel = dict((value, key) for key, value in label.items())

    predicts = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size, verbose=2)
    print(predicts)
    predict_class = np.argmax(predicts, axis=1)
    print(predict_class)
    errors = np.where(predict_class != truth)[0]
    print("No of errors = {}/{}".format(len(errors),validation_generator.samples))

    plt.rcParams["axes.grid"] = True
    plt.rcParams.update({'font.size': 20})

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(truth,predict_class)
    labels = []
    for k,v in indexlabel.items():
        labels.append(v)
    import itertools
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

    plot_confusion_matrix(cm, classes=labels, title='Confusion Matrix')

    #In[]:
    print("\n")
    print("\n")
    print("classification_report\n")

    print(classification_report(truth, predict_class))
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
    model.save(" Model_cross")
    i+=1


#In[]
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from keras.applications.imagenet_utils import decode_predictions

model_builder = keras.applications.densenet.DenseNet121

img_size = (224, 224)
preprocess_input = keras.applications.densenet.preprocess_input
decode_predictions = keras.applications.densenet.decode_predictions

last_conv_layer_name = "conv5_block16_concat"

img_path = "0GENLS64M134.jpg"
#plt.imshow(img_path)
display(Image(img_path))

def get_img_array(img_path, size):
    
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

img_array = preprocess_input(get_img_array(img_path, size=img_size))

model = model = model_builder(weights=model.load_weights('./DeepLearning/cross_validation/checkpoint'))

model.layers[-1].activation = None

preds = model.predict(img_array)

heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)

    display(Image(cam_path))
