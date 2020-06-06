from pathlib import Path
import numpy as np
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt

pixPerSec = 75 #number of pixels within one second of audio
augment_factor = 25 #how many seconds to shift the reading frame
clipLength = 5 #length, in seconds, of each audio clip
timeOffset = 62
freqBound = 160
frame_shift = (pixPerSec*clipLength)/augment_factor

##Makedir for processed data
finished_dir = Path.cwd().joinpath('Data_Set', 'Clipped_Data')
finished_dir.mkdir(exist_ok = 'true')

##Import files
spec_dir = Path.cwd().joinpath('Data_Set', 'Raw_Data')
for species in spec_dir.iterdir(): #iterates through the species folders
	
	finished_spec = Path.joinpath(finished_dir, species.name)#Makedir for species
	finished_spec.mkdir(exist_ok = 'true')

	spec_img = Path(species)
	img_num = 1 #used for labeling processed images

	for image in spec_img.iterdir(): #iterates through the images within each species
		img = Image.open(image)
		img = img.convert("L")
		l = img.size
		img = img.crop((timeOffset,0, l[0], freqBound)) #crops the image to just the spectrogram

		shift_num = 0

		while shift_num < augment_factor:
			initialOffset = (frame_shift*shift_num)
			frame_num = 1
			newFrameEnd = (initialOffset+pixPerSec*clipLength)*frame_num
			new_time_offset = initialOffset

			while newFrameEnd <= l[0]:
				newImg = img.crop((new_time_offset,0, newFrameEnd, freqBound))
				img_name = species.stem + "_" + str(img_num) + "_" + str(shift_num) + "_" + str(frame_num) + ".png"
				img_path = Path.joinpath(finished_spec, img_name)
				newImg.save(img_path)
				new_time_offset = newFrameEnd
				frame_num += 1
				newFrameEnd = (new_time_offset+pixPerSec*clipLength)

			if (l[0] - newFrameEnd) >= (2*pixPerSec):
				newImg = ImageOps.pad(img.crop((new_time_offset,0, newFrameEnd, freqBound)), ((pixPerSec*clipLength),freqBound), color = (255, 255, 255), centering = (0,0))
				img_name = species.stem + "_" + str(img_num) + "_" + str(shift_num) + "_" + str(frame_num) + ".png"
				img_path = Path.joinpath(finished_spec, img_name)
				newImg.save(img_path)

			shift_num += 1

		img_num += 1
		
