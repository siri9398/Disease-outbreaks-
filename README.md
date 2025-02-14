{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "hello world\n"
     ]
    }
   ],
   "source": [
    "print(\"hello world\")\n",
    "# alt + Enter "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "pip 24.3.1 from d:\\Users\\Lib\\site-packages\\pip (python 3.12)\n",
      "\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "%pip --version"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%pip install tensorflow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Current Dir  c:\\Users\\jayra\\OneDrive\\Documents\\potato-disease-project\n"
     ]
    }
   ],
   "source": [
    "import os \n",
    "print(\"Current Dir \",os.getcwd())\n",
    "curr_dir = os.getcwd()\n",
    "train_path = os.path.join(curr_dir,\"datasets\",\"Train\")\n",
    "test_path=os.path.join(curr_dir,\"datasets\",\"Test\")\n",
    "valid_path= os.path.join(curr_dir,\"datasets\",\"Valid\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 900 files belonging to 3 classes.\n"
     ]
    }
   ],
   "source": [
    "training_set = tf.keras.utils.image_dataset_from_directory(\n",
    "    train_path,\n",
    "    labels=\"inferred\",\n",
    "    label_mode=\"categorical\",\n",
    "    color_mode=\"rgb\",\n",
    "    image_size=(128, 128),\n",
    "    shuffle=True,\n",
    "     interpolation=\"bilinear\",\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "training_set.class_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 300 files belonging to 3 classes.\n"
     ]
    }
   ],
   "source": [
    "validation_set = tf.keras.utils.image_dataset_from_directory(\n",
    "    valid_path,\n",
    "    labels=\"inferred\",\n",
    "    label_mode=\"categorical\",\n",
    "    color_mode=\"rgb\",\n",
    "    image_size=(128, 128),\n",
    "    shuffle=True,\n",
    "     interpolation=\"bilinear\",\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "d:\\Users\\Lib\\site-packages\\keras\\src\\layers\\convolutional\\base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
     ]
    }
   ],
   "source": [
    "cnn = tf.keras.models.Sequential()\n",
    "\n",
    "cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))\n",
    "cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))\n",
    "cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))\n",
    "\n",
    "cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))\n",
    "cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))\n",
    "cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))\n",
    "\n",
    "cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))\n",
    "cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu'))\n",
    "cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))\n",
    "\n",
    "cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))\n",
    "cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,activation='relu'))\n",
    "cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))\n",
    "\n",
    "cnn.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))\n",
    "cnn.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,activation='relu'))\n",
    "cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))\n",
    "\n",
    "cnn.add(tf.keras.layers.Dropout(0.25))\n",
    "\n",
    "cnn.add(tf.keras.layers.Flatten())\n",
    "cnn.add(tf.keras.layers.Dense(units=1500,activation='relu'))\n",
    "cnn.add(tf.keras.layers.Dropout(0.4))\n",
    "\n",
    "cnn.add(tf.keras.layers.Dense(units=3,activation='softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "cnn.compile(optimizer=tf.keras.optimizers.Adam(\n",
    "    learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ conv2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)   │           <span style=\"color: #00af00; text-decoration-color: #00af00\">896</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">126</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">126</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)   │         <span style=\"color: #00af00; text-decoration-color: #00af00\">9,248</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ max_pooling2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">63</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">63</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)     │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">63</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">63</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)     │        <span style=\"color: #00af00; text-decoration-color: #00af00\">18,496</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">61</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">61</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)     │        <span style=\"color: #00af00; text-decoration-color: #00af00\">36,928</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ max_pooling2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">30</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">30</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)     │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_4 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">30</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">30</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)    │        <span style=\"color: #00af00; text-decoration-color: #00af00\">73,856</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_5 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)    │       <span style=\"color: #00af00; text-decoration-color: #00af00\">147,584</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ max_pooling2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)    │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_6 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │       <span style=\"color: #00af00; text-decoration-color: #00af00\">295,168</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_7 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">12</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">12</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)    │       <span style=\"color: #00af00; text-decoration-color: #00af00\">590,080</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ max_pooling2d_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_8 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)      │     <span style=\"color: #00af00; text-decoration-color: #00af00\">1,180,160</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_9 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)      │     <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ max_pooling2d_4 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ flatten (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2048</span>)           │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                   │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1500</span>)           │     <span style=\"color: #00af00; text-decoration-color: #00af00\">3,073,500</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)             │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1500</span>)           │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">3</span>)              │         <span style=\"color: #00af00; text-decoration-color: #00af00\">4,503</span> │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ conv2d (\u001b[38;5;33mConv2D\u001b[0m)                 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m128\u001b[0m, \u001b[38;5;34m128\u001b[0m, \u001b[38;5;34m32\u001b[0m)   │           \u001b[38;5;34m896\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_1 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m126\u001b[0m, \u001b[38;5;34m126\u001b[0m, \u001b[38;5;34m32\u001b[0m)   │         \u001b[38;5;34m9,248\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ max_pooling2d (\u001b[38;5;33mMaxPooling2D\u001b[0m)    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m63\u001b[0m, \u001b[38;5;34m63\u001b[0m, \u001b[38;5;34m32\u001b[0m)     │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_2 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m63\u001b[0m, \u001b[38;5;34m63\u001b[0m, \u001b[38;5;34m64\u001b[0m)     │        \u001b[38;5;34m18,496\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_3 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m61\u001b[0m, \u001b[38;5;34m61\u001b[0m, \u001b[38;5;34m64\u001b[0m)     │        \u001b[38;5;34m36,928\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ max_pooling2d_1 (\u001b[38;5;33mMaxPooling2D\u001b[0m)  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m30\u001b[0m, \u001b[38;5;34m30\u001b[0m, \u001b[38;5;34m64\u001b[0m)     │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_4 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m30\u001b[0m, \u001b[38;5;34m30\u001b[0m, \u001b[38;5;34m128\u001b[0m)    │        \u001b[38;5;34m73,856\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_5 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m128\u001b[0m)    │       \u001b[38;5;34m147,584\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ max_pooling2d_2 (\u001b[38;5;33mMaxPooling2D\u001b[0m)  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m128\u001b[0m)    │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_6 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │       \u001b[38;5;34m295,168\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_7 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m12\u001b[0m, \u001b[38;5;34m12\u001b[0m, \u001b[38;5;34m256\u001b[0m)    │       \u001b[38;5;34m590,080\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ max_pooling2d_3 (\u001b[38;5;33mMaxPooling2D\u001b[0m)  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m256\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_8 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m512\u001b[0m)      │     \u001b[38;5;34m1,180,160\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_9 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m512\u001b[0m)      │     \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ max_pooling2d_4 (\u001b[38;5;33mMaxPooling2D\u001b[0m)  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2\u001b[0m, \u001b[38;5;34m2\u001b[0m, \u001b[38;5;34m512\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout (\u001b[38;5;33mDropout\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2\u001b[0m, \u001b[38;5;34m2\u001b[0m, \u001b[38;5;34m512\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ flatten (\u001b[38;5;33mFlatten\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2048\u001b[0m)           │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense (\u001b[38;5;33mDense\u001b[0m)                   │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1500\u001b[0m)           │     \u001b[38;5;34m3,073,500\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout_1 (\u001b[38;5;33mDropout\u001b[0m)             │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1500\u001b[0m)           │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m3\u001b[0m)              │         \u001b[38;5;34m4,503\u001b[0m │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">7,790,227</span> (29.72 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m7,790,227\u001b[0m (29.72 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">7,790,227</span> (29.72 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m7,790,227\u001b[0m (29.72 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "cnn.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "\u001b[1m29/29\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 446ms/step - accuracy: 0.3475 - loss: 1.4849 - val_accuracy: 0.4500 - val_loss: 1.0618\n",
      "Epoch 2/10\n",
      "\u001b[1m29/29\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m12s\u001b[0m 428ms/step - accuracy: 0.5390 - loss: 0.9281 - val_accuracy: 0.6967 - val_loss: 0.6125\n",
      "Epoch 3/10\n",
      "\u001b[1m29/29\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m12s\u001b[0m 422ms/step - accuracy: 0.7214 - loss: 0.6263 - val_accuracy: 0.7567 - val_loss: 0.4926\n",
      "Epoch 4/10\n",
      "\u001b[1m29/29\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m12s\u001b[0m 426ms/step - accuracy: 0.7783 - loss: 0.4654 - val_accuracy: 0.7267 - val_loss: 0.6456\n",
      "Epoch 5/10\n",
      "\u001b[1m29/29\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m12s\u001b[0m 425ms/step - accuracy: 0.8110 - loss: 0.4781 - val_accuracy: 0.8800 - val_loss: 0.2691\n",
      "Epoch 6/10\n",
      "\u001b[1m29/29\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m13s\u001b[0m 432ms/step - accuracy: 0.9057 - loss: 0.2284 - val_accuracy: 0.8767 - val_loss: 0.2833\n",
      "Epoch 7/10\n",
      "\u001b[1m29/29\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m13s\u001b[0m 430ms/step - accuracy: 0.9177 - loss: 0.2252 - val_accuracy: 0.9000 - val_loss: 0.2440\n",
      "Epoch 8/10\n",
      "\u001b[1m29/29\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m13s\u001b[0m 440ms/step - accuracy: 0.9288 - loss: 0.2033 - val_accuracy: 0.8967 - val_loss: 0.2889\n",
      "Epoch 9/10\n",
      "\u001b[1m29/29\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 509ms/step - accuracy: 0.9085 - loss: 0.2582 - val_accuracy: 0.9300 - val_loss: 0.1715\n",
      "Epoch 10/10\n",
      "\u001b[1m29/29\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m12s\u001b[0m 421ms/step - accuracy: 0.9652 - loss: 0.0901 - val_accuracy: 0.9300 - val_loss: 0.1929\n"
     ]
    }
   ],
   "source": [
    "training_history = cnn.fit(x=training_set,validation_data=validation_set,epochs=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
