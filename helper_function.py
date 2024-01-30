{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMhJnCuNAy1FQfHHzC6+jkO",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Abhilash-Bee/DeepLearningCertification/blob/main/helper_function.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "metadata": {
        "id": "OmBkd8I6kvpV"
      },
      "outputs": [],
      "source": [
        "# Importing Dependencies\n",
        "from tensorflow.keras.utils import get_file\n",
        "import matplotlib.pyplot as plt\n",
        "import tensorflow as tf\n",
        "\n",
        "\n",
        "# Dowload and unzip the file from `url`\n",
        "def unzip_file(url: str) -> tuple:\n",
        "  \"\"\"\n",
        "  Downloads and unzips the file.\n",
        "\n",
        "  Args:\n",
        "  url - url of the file to be downloaded and unzip\n",
        "\n",
        "  Returns:\n",
        "  tuple - (train path, test path) training folder path and testing folder path\n",
        "  \"\"\"\n",
        "\n",
        "  path = get_file(origin=url, extract=True)\n",
        "  return (path[:-4]+'/train/', path[:-4]+'/test/')\n",
        "\n",
        "\n",
        "\n",
        "# Plot loss and accuracy curve\n",
        "def plot_loss_accuracy_curve(history, epoch):\n",
        "  \"\"\"\n",
        "  Plots the loss and accuracy curve on training and validation history.\n",
        "\n",
        "  Args:\n",
        "  history - history of the model\n",
        "\n",
        "  Returns:\n",
        "  Plots the loss and accuracy curve of training and validation history.\n",
        "  \"\"\"\n",
        "\n",
        "  fig, ax = plt.subplots(2, 1, figsize=(8, 17), sharex=True)\n",
        "\n",
        "  loss = [history.history['loss'], history.history['val_loss']]\n",
        "  accuracy = [history.history['accuracy'], history.history['val_accuracy']]\n",
        "  loss_accuracy = [loss, accuracy]\n",
        "  labels = ['Loss', 'Accuracy']\n",
        "\n",
        "  for i in range(2):\n",
        "    ax[i].set_title('Loss Vs Epoch Curve')\n",
        "    ax[i].plot(tf.range(1, epoch+1), loss_accuracy[i][0])\n",
        "    ax[i].plot(tf.range(1, epoch+1), loss_accuracy[i][1])\n",
        "    ax[i].set_ylabel(labels[i])\n",
        "\n",
        "  ax[1].set_xlabel('Epoch')\n",
        "\n",
        "\n",
        "\n",
        "#"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "vPaL-SQKnwip"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "files = get_file(origin=\"https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip\", extract=True)\n",
        "files"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 67
        },
        "id": "6c_PY_bbmW7E",
        "outputId": "d14be0b1-8d2d-4d5e-9563-739ed11c2ea3"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading data from https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip\n",
            "168546183/168546183 [==============================] - 1s 0us/step\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'/root/.keras/datasets/10_food_classes_10_percent.zip'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!ls /root/.keras/datasets/10_food_classes_10_percent/train/"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cIURl4VWpIbJ",
        "outputId": "4bd9bbec-9875-4061-a05c-9419c1b20c00"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "chicken_curry  fried_rice      hamburger  pizza  steak\n",
            "chicken_wings  grilled_salmon  ice_cream  ramen  sushi\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!ls /root/.keras/datasets/"
      ],
      "metadata": {
        "id": "XCGU-Bvmnhy9"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!rm -rf /root/.keras/datasets/10_food_classes_10_percent"
      ],
      "metadata": {
        "id": "GwD7aHKinAZF"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(files[:-4])"
      ],
      "metadata": {
        "id": "QkW6Zy_Rmnxx",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "218b40e4-6894-410c-b194-54934b72722f"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/root/.keras/datasets/10_food_classes_10_percent\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "type(files[:-4])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2GLcUNuEpkdN",
        "outputId": "d14eb2dd-91fb-4ab0-e7ed-c6e1f0e33dc1"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "str"
            ]
          },
          "metadata": {},
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "QUQx_wWipnUW"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}