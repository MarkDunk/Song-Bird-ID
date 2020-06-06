##Based on https://github.com/kalaspuffar/tensorflow-data/blob/master/create_dataset.py
from pathlib import Path
import numpy as np
from PIL import Image
from random import shuffle
import tensorflow as tf
import glob
import Prepro

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def datasetGen(filename, img_glob, labels):
	with tf.io.TFRecordWriter(filename) as writer:
		for i in range(len(img_glob)):
			img = Image.open(img_glob[i])
			img = img.resize(image_size)

			label = labels[i]

			feature = {
				'image_raw': _bytes_feature(img.tobytes()),
				'label': _int64_feature(label)
			}

			example = tf.train.Example(features=tf.train.Features(feature=feature))

			writer.write(example.SerializeToString())

		writer.close()


species_list = {
	"American Goldfinch" : 0,
	"American Robin" : 1,
	"Baltimore Oriole" : 2,
	"Black-capped Chickadee" : 3,
	"Blue Jay" : 4,
	"Brown-headed Cowbird" : 5,
	"Common Grackle" : 6,
	"Common Starling" : 7,
	"Dark-eyed Junco" : 8,
	"House Finch" : 9,
	"House Sparrow" : 10,
	"Indigo Bunting" : 11,
	"Northern Cardinal" : 12,
	"Northern Flicker" : 13,
	"Purple Martin" : 14
}

image_size = (192,80)

img_path = 'Data_Set/Clipped_Data/*/*.png'
img_glob =  glob.glob(img_path)
species_path = Path().glob(img_path)
labels = [species_list[species.parent.name]for species in species_path]

# print(labels)

#print(species_path)

data = list(zip(img_glob, labels))
shuffle(data)
img_glob, labels = zip(*data)

train_img = img_glob[0:int(0.6*len(img_glob))]
train_labels = labels[0:int(0.6*len(labels))]
val_img = img_glob[int(0.6*len(img_glob)):int(0.8*len(img_glob))]
val_labels = labels[int(0.6*len(labels)):int(0.8*len(labels))]
test_img = img_glob[int(0.8*len(img_glob)):]
test_labels = labels[int(0.8*len(labels)):]

datasetGen('Data_Set/train.tfrecords', train_img, train_labels)
datasetGen('Data_Set/val.tfrecords', val_img, val_labels)
datasetGen('Data_Set/test.tfrecords', test_img, test_labels)