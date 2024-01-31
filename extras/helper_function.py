# Common Dependencies
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# Dowload and unzip the file from `url`
def unzip_file(url: str) -> str:
  """
  Downloads and unzips the file.

  Args:
  url - url of the file to be downloaded and unzip

  Returns:
  Path of dataset directory
  """

  path = tf.keras.utils.get_file(origin=url, extract=True)
  print(f'{path[-4]} has been successfully extracted.')
  return path[:-4]



# Plot loss and accuracy curve
def plot_loss_accuracy_curve(history, figsize=(7, 10), savefig=False) -> None:
  """
  Plots the loss and accuracy curve on training and validation history.

  Args:
  history - history of the model
  figsize - defaults to `(8, 17)`
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



# Importing Dependencies
import datetime

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
  callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f'Saving the tensorboard callbacks in {log_dir}')
  return callback
