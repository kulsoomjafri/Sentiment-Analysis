import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
#import split_folders
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, AveragePooling2D
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import load_img, img_to_array
import time
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import itertools
AUTOTUNE = tf.data.experimental.AUTOTUNE
import PIL.Image
import urllib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_HEIGHT = 48
IMG_WIDTH = 48

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES
  
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label 

def show_batch(image_batch, label_batch,title):
  plt.figure(figsize=(10,10))
  plt.suptitle(title, fontsize=16)

  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')
def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)
  

cwd_dir = os.getcwd()#get the current directory
#unzipped the filed named as image_detection.zip to folder train
data = os.path.join(cwd_dir, 'face-expression-recognition-dataset')
images = os.path.join(data, 'images')
train_dir = os.path.join(images, 'train')
val_dir = os.path.join(images, 'validation')
test_dir = os.path.join(images, 'test')

train_angry_dir = os.path.join(train_dir, 'angry') 
train_depressed_dir = os.path.join(train_dir, 'depressed') 
train_disgust_dir = os.path.join(train_dir, 'disgust')  
train_happy_dir = os.path.join(train_dir, 'happy') 
train_neutral_dir = os.path.join(train_dir, 'neutral') 
train_surprise_dir = os.path.join(train_dir, 'surprise') 

val_angry_dir = os.path.join(val_dir, 'angry')
val_depressed_dir = os.path.join(val_dir, 'depressed')
val_disgust_dir = os.path.join(val_dir, 'disgust')
val_happy_dir = os.path.join(val_dir, 'happy')
val_neutral_dir = os.path.join(val_dir, 'neutral')
val_surprise_dir = os.path.join(val_dir, 'surprise')


num_angry_tr = len(os.listdir(train_angry_dir))
num_depressed_tr = len(os.listdir(train_depressed_dir))
num_disgust_tr = len(os.listdir(train_disgust_dir))
num_happy_tr = len(os.listdir(train_happy_dir))
num_neutral_tr = len(os.listdir(train_neutral_dir))
num_surprise_tr = len(os.listdir(train_surprise_dir))

num_angry_val = len(os.listdir(val_angry_dir))
num_depressed_val = len(os.listdir(val_depressed_dir))
num_disgust_val = len(os.listdir(val_disgust_dir))
num_happy_val = len(os.listdir(val_happy_dir))
num_neutral_val = len(os.listdir(val_neutral_dir))
num_surprise_val = len(os.listdir(val_surprise_dir))

total_train = num_angry_tr + num_depressed_tr + num_disgust_tr + num_happy_tr + num_neutral_tr + num_surprise_tr
total_val = num_angry_val + num_depressed_val + num_disgust_val + num_happy_val + num_neutral_val + num_surprise_val

print('total training angry images:', num_angry_tr)
print('total training depressed images:', num_depressed_tr)
print('total training disgust images:', num_disgust_tr)
print('total training happy images:', num_happy_tr)
print('total training neutral images:', num_neutral_tr)
print('total training surprise images:', num_surprise_tr)
print (sep= "\n")
print('total validation angry images:', num_angry_val)
print('total validation depressed images:', num_depressed_val)
print('total validation disgust images:', num_disgust_val)
print('total validation happy images:', num_happy_val)
print('total validation neutral images:', num_neutral_val)
print('total validation surprise images:', num_surprise_val)
print (sep= "\n")
print("Total training images:", total_train)
print("Total validation images:", total_val)

batch_size = 500
epochs = 50
IMG_HEIGHT = 48
IMG_WIDTH = 48

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
print (sep= "\n")
train_data_gen = train_image_generator.flow_from_directory(batch_size=total_train,
                                                           directory=train_dir,
                                                           color_mode="grayscale",
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')
data_dir = pathlib.Path(train_dir)  #creating glob
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if (item.name != "LICENSE.txt" and item.name != "desktop.ini")])
print ("The training classes are", CLASS_NAMES)

val_data_gen = validation_image_generator.flow_from_directory(batch_size=total_val,
                                                              directory=val_dir,
                                                              color_mode="grayscale",
                                                              shuffle=False,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')
                                                              
data_dir = pathlib.Path(val_dir)  #creating glob
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if (item.name != "LICENSE.txt" and item.name != "desktop.ini")])
print ("The validation classes are", CLASS_NAMES)

sample_training_images, _ = next(train_data_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(2, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        img2=np.squeeze(img)
        ax.imshow(img2, cmap='gray')
        ax.axis('off')
    plt.tight_layout()

plotImages(sample_training_images[:20])

image_batch, label_batch = next(train_data_gen)
# aug_image_batch, aug_label_batch = next(augmented_data_gen)
val_batch, val_label_batch = next(val_data_gen)
# plotImages(aug_image_batch[:20])
# print(image_batch[:2])

# print(augmented_images[:2])
# x=np.concatenate((image_batch, aug_image_batch), axis= None)
# y=np.concatenate((label_batch, aug_label_batch), axis= None)
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.optimizers import *
model = Sequential([
        Conv2D(16, 5, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,1)),
        AveragePooling2D((2,2)),
        Dropout(0.25),
        Conv2D(64, (5,5),  padding='same', activation='relu'),
        Conv2D(128,(5,5),  padding='same', activation='relu'),
        Dropout(0.25),
        Conv2D(512,(3,3) ,padding='same', activation='relu'),
        Conv2D(512,(3,3) ,padding='same', activation='relu'),
        AveragePooling2D((2,2)),
        Dropout(0.25),
        Flatten(),
        # Dense(128, activation='relu'),
        # Dropout(0.25),
        Dense(512, activation='relu'),
        Dense(6,activation=tf.nn.softmax)
    ])

opt = Adam(lr=0.001)  
model.compile(optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.summary()

image_batch, label_batch = next(train_data_gen)
# aug_image_batch, aug_label_batch = next(augmented_data_gen)
val_batch, val_label_batch = next(val_data_gen)
X_train=image_batch
y_train=label_batch
X_valid=val_batch
y_valid=val_label_batch
#data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
)
train_datagen.fit(X_train)

batch_size=128
epochs=50

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00005,
    patience=11,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1,
)
callbacks = [early_stopping,lr_scheduler]
# history = model.fit(x=image_batch,y=label_batch, batch_size=1000 , validation_data=[val_batch, val_label_batch], verbose=1, epochs=30)
history = model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_valid, y_valid),
    steps_per_epoch=len(X_train) / batch_size,
    epochs=epochs,
    callbacks=callbacks
)


plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
# plt.show()
#<Testing>

image_gen_test = ImageDataGenerator(
                    rescale=1./255,
                    )
test_data_gen = image_gen_test.flow_from_directory(batch_size=total_train,
                                                     directory=train_dir,
                                                     color_mode="grayscale",
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical')


test_batch,_ = next(test_data_gen)
predictor=test_batch[:36]
predictions = model.predict(predictor)
b = tf.math.argmax(input = predictions.transpose())
c = tf.keras.backend.eval(b)
plt.figure(figsize=(10,10))
plt.suptitle('Images Predicted from Test data', fontsize=15)
for n in range(36):
    ax = plt.subplot(6,6,n+1)
    img2=np.squeeze(predictor[n])
    plt.imshow(img2,cmap='gray')
    plt.title(CLASS_NAMES[c[n]])
    plt.axis('off')
    
plt.show()

# #confusion_matrix
# predictor=test_batch
# predictions= model.predict(predictor)
# b = tf.math.argmax(input = predictions.transpose())
# predictions = tf.keras.backend.eval(b)
# print("Size of test images is")
# print(len(predictions_all))


# true_label = tf.math.argmax(input =(test_label_batch.numpy()*1).transpose())
# cnf_matrix = confusion_matrix(true_label, predictions)

# np.set_printoptions(precision=2)

# # plot normalized confusion matrix
# plot_confusion_matrix(cnf_matrix, classes=CLASS_NAMES, title='Normalized confusion matrix')

#<\Testing>
