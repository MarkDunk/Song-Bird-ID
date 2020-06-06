import tensorflow as tf
import numpy as np
import os, datetime
from pathlib import Path

img_shape = (80, 192, 1)
num_birb = 15
epoch = 20
batch_size = 256
buffer_size = 10000

image_feature = {
	'image_raw': tf.io.FixedLenFeature([], tf.string),
	'label': tf.io.FixedLenFeature([], tf.int64)
}

def parse_data(example):
	parse = tf.io.parse_single_example(example, image_feature)
	image = tf.io.decode_raw(parse["image_raw"], tf.uint8)
	image = tf.cast(image, tf.float32)
	image = tf.reshape(image, img_shape)
	image = image/255

	label = tf.cast(parse["label"], tf.int32)

	return image, label

raw_train_data = tf.data.TFRecordDataset('train.tfrecords')
raw_val_data = tf.data.TFRecordDataset('val.tfrecords')

raw_train_data = raw_train_data.shuffle(buffer_size, reshuffle_each_iteration=True)
train_data = raw_train_data.map(parse_data).batch(batch_size)
val_data = raw_val_data.map(parse_data).batch(batch_size)

# network
model = tf.keras.Sequential([
	tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape =img_shape, padding = 'same'),
	tf.keras.layers.MaxPooling2D(2,2),

	tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', padding = 'same'),
	tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
	tf.keras.layers.MaxPooling2D(2,2),

	tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
	tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'),	
	tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same'),
	tf.keras.layers.MaxPooling2D(2,2),

	tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'),
	tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same'),
	tf.keras.layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same'),
	tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', padding = 'same'),
	tf.keras.layers.MaxPooling2D(2,2),

	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(64, activation = 'relu'),
	tf.keras.layers.Dense(num_birb, activation = 'softmax')
])

model.compile(
	optimizer= 'adam', 
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'] 
	)

logdir = os.path.join("TB_log_dir", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

checkpoint_path = "checkpoints/cp{epoch}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_only_best=True,
    verbose=0,
    save_freq=(1000)
)

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint != None:
    model.load_weights(latest_checkpoint)

model.fit(
	train_data,
	epochs=epoch,
	validation_data=val_data,
    validation_steps=200,
	callbacks=[checkpoint_callback, tensorboard_callback]
)