### Before running, you may need to produce a series of images from the video using the command:    
ffmpeg -i input_video -vf fps=60 output%05d.jpg    
ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 output.wav


### Files:
#### crop.py:   
Generating the inputs for iTracker, including the images of both eyes, faces and face grid for each image, please go to https://github.com/qiexing/face-landmark-localization to compile this package follow the instructions, and then copy 'crop.py' into the installation directory of 'face-landmark-localization' package. After modifying the attiributes such as input directory, you can run this file directory to generate features for iTracker.
	
#### Calibration.py:    
Performing calibration and store the captured images.
	
#### Synchronize.ipynb:    
Synchronize data sequences from Tobii and iTracker.
	
#### iTracker.ipynb:    
Generating frames using iTracker, producing saliency maps and eye gaze gaussian images, implement the time-series models.
	
#### CNN.ipynb:    
The settings for the CNN regression model.
