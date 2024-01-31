# Common Dependencies
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
import datetime



# Dowload and unzip the file from `url`
def get_file(url: str) -> str:
  """
  Downloads and unzips the file.

  Args:
  url - url of the file to be downloaded and unzip

  Returns:
  Path of dataset directory
  """

  path = tf.keras.utils.get_file(origin=url, extract=True)
  print(f'{path[:-4]} has been successfully extracted.')
  return path[:-4]



# Plot the images from the given folder
def plot_images(directory: str, class_names: list, _from='train', img_shape=224, figsize=(10, 6)) -> None:
  """
  Selects random images from the directory of class_names (sub-directory) and
  plot 6 random images from random classes of the directory.

  Args:
  directory - path of the dataset (train or test)
  class_names - labels (categorical or binary)
  _from - defaults to 'train', can be altered to 'validation'/'test'
  img_shape - defaults to (224, 224)
  figsize - defaults to (10, 6)
  """

  rand_class_names = [random.choice(class_names) for _ in range(6)]
  rand_imgs = [random.choice(os.listdir(directory + '/' + _from + '/' + class_name + '/')) for class_name in rand_class_names]

  fig, ax = plt.subplots(2, 3, figsize=figsize)

  k = 0
  for i in range(2):
    for j in range(3):
      class_name = rand_class_names[k]
      img_name = rand_imgs[k]
      img_path = directory + '/' + _from + '/' + class_name + '/' + img_name

      ax[i][j].set_title(class_name)
      img = tf.io.read_file(img_path)
      img = tf.image.decode_jpeg(img)
      img = tf.image.resize(img, [img_shape, img_shape])
      img = img / 255.
      ax[i][j].imshow(img)
      ax[i][j].set_xticks([])
      ax[i][j].set_yticks([])

      k = k + 1



# Plot the images of the model which got trained
def pred_and_plot(model, directory: str, class_names: list, _from='test', img_shape=224, figsize=(10, 6)):
  """
  Selects random images from the directory of class_names (sub-directory) and
  plot 6 random images from random classes of the directory with Original
  classname and predicited classname.

  Args:
  directory - path of the dataset (train or test)
  class_names - labels (categorical or binary)
  _from - defaults to 'test', can be altered to 'train'/validation'
  img_shape - defaults to (224, 224)
  figsize - defaults to (10, 6)
  """

  rand_class_names = [random.choice(class_names) for _ in range(6)]
  rand_imgs = [random.choice(os.listdir(directory + '/' + _from + '/' + class_name + '/')) for class_name in rand_class_names]

  fig, ax = plt.subplots(2, 3, figsize=figsize)

  k = 0
  for i in range(2):
    for j in range(3):
      class_name = rand_class_names[k]
      img_name = rand_imgs[k]
      img_path = directory + '/' + _from + '/' + class_name + '/' + img_name

      img = tf.io.read_file(img_path)
      img = tf.image.decode_jpeg(img)
      img = tf.image.resize(img, [img_shape, img_shape])
      img = img / 255.
      ax[i][j].imshow(img)

      y_prob = model.predict(tf.expand_dims(img, axis=0))
      if len(y_prob[0]) > 1:
        y_pred = tf.argmax(y_prob[0])
      else:
        y_pred = tf.where(y_prob < 0.5, 0, 1)

      if y_pred == class_name:
        color = 'green'
      else:
        color = 'red'
      
      ax[i][j].set_title(f'Original: {class_names[class_name]}\nPredicted: {class_names[y_pred]}', color=color)

      k = k + 1



# Plot loss and accuracy curve
def plot_loss_accuracy_curve(history, figsize=(7, 10), savefig=False) -> None:
  """
  Plots the loss and accuracy curve on training and validation history.

  Args:
  history - history of the model
  figsize - defaults to `(7, 10)`
  savefig - defaults to `False`, if `True`, saves the image as `.png`
  """

  fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)

  loss = [history.history['loss'], history.history['val_loss']]
  accuracy = [history.history['accuracy'], history.history['val_accuracy']]
  loss_accuracy = [loss, accuracy]
  labels = ['Loss', 'Accuracy']
  epoch = tf.range(1, len(history.history['loss']) + 1)

  for i in range(2):
    ax[i].set_title(labels[i] + ' Vs Epoch Curve')
    ax[i].plot(epoch, loss_accuracy[i][0], label = 'Training ' + labels[i])
    ax[i].plot(epoch, loss_accuracy[i][1], label = 'Validation ' + labels[i])
    ax[i].set_ylabel(labels[i])
    if i == 0:
      ax[i].legend(loc='upper right')
    else:
      ax[i].legend(loc='upper left')

  ax[1].set_xlabel('Epoch')

  if savefig:
    fig.savefig(f"plot_loss_accuracy_curve_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")



# Comparing histories of model
def compare_histories(original_history, new_history, figsize=(7, 10), savefig=False) -> None:
  """
  Plots the loss and accuracy curve on training and validation of original history
  and new history.

  Args:
  original_history - history of the original model
  new_history - history of the new model
  figsize - defaults to `(7, 10)`
  savefig - defaults to `False`, if `True`, saves the image as `.png`
  """

  tot_loss = original_history.history['loss'] + new_history.history['loss']
  tot_val_loss = original_history.history['val_loss'] + new_history.history['val_loss']

  tot_accuracy = original_history.history['accuracy'] + new_history.history['accuracy']
  tot_val_accuracy = original_history.history['val_accuracy'] + new_history.history['val_accuracy']

  labels = ['Loss', 'Accuracy']
  loss_accuracy = [(tot_loss, tot_val_loss), (tot_accuracy, tot_val_accuracy)]

  initial_epoch = len(original_history.history['loss'])

  fig, ax = plt.subplots(2, 1, figsize=figsize)

  for i in range(2):
    ax[i].set(ylabel=labels[i],
              title=labels[i]+' Vs Epoch curve')
    ax[i].plot(loss_accuracy[i][0], label='Training ' + labels[i])
    ax[i].plot(loss_accuracy[i][1], label='Validation ' + labels[i])
    ax[i].plot([initial_epoch - 1, initial_epoch - 1], plt.ylim(), label='Start Fine Tuning')
    if i == 0:
      ax[i].legend(loc='upper right')
    else:
      ax[i].legend(loc='upper left')

  ax[1].set_xlabel='Epoch'

  if savefig:
    fig.savefig(f"plot_compare_history_loss_accuracy_curve_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")



# Importing Dependencies
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, figsize=(10, 10), class_names=None) -> None:
  """
  Plots confusion matrix on the true and predicted label.

  Args:
  y_true - Acutal (True) Label
  y_pred - Predicted Label
  figsize - Defaults to `(10, 10)`
  class_names - Defaults to `None`, provide class_names to change x and y labels
  """

  cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
  fig, ax = plt.subplots(figsize=figsize)
  disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
  disp.plot(ax=ax, xticks_rotation='vertical', cmap=plt.cm.Blues)



# Importing Dependencies
import os

# Walk through the directory
def walk_through_directory(filepath: str) -> None:
  """
  Provides number of folders and number of filenames along with filepath by
  running recursively into the subdirectories of filepath.

  Args:
  filepath - path of the directory
  """

  for dirpath, dirnames, filenames in os.walk(filepath):
    print(f"There are {len(dirnames)} folders and {len(filenames)} in this '{dirpath}' directory path.")



# Create the tensorboard callback
def tensorboard_callbacks(directory: str, experiment_name: str) -> object:
  """
  Creates a tensorboard callback and provides message if the tensorboard callback
  is successfully saved in the path.

  Args:
  directory - folder name of the tensorboard callback
  experiment_name - sub-folder inside the directory of the current experiment

  Returns:
  Tensorboard object
  """

  log_dir = directory + '/' + experiment_name + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, )
  print(f'Saving the tensorboard callbacks in {log_dir}')
  return callback



# Create the tensorflow ModelCheckpoint
def tensorflow_modelcheckpoint(directory: str, experiment_name: str):
  """
  Creates a model checkpoint and saves in the provided directory with experiment_name
  as sub-directory with another sub-directory as datetime.

  Args:
  directory - folder name of the tensorboard model checkpoint
  experiment_name - sub-folder inside the directory of the current experiment

  Returns:
  ModelCheckpoint object
  """

  filepath = directory + '/' + experiment_name + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  return tf.keras.callbacks.ModelCheckpoint(filepath=filepath, 
                                                verbose=1, 
                                                save_best_only=True, 
                                                save_weights_only=True)
