import time
import cv2
import os
import pygame,sys
from pygame import *
import time
WIDTH = 1280
HEIGHT =1024
WHITE = (255,255,255) #RGB
BLACK = (0,0,0) #RGB
RED=(255,0,0)
pygame.init()
screen = display.set_mode((WIDTH,HEIGHT), pygame.FULLSCREEN)
display.set_caption("Name of Application")
screen.fill(BLACK)
display.update()
timer = pygame.time.Clock()
radius = 12
previous=(0,0)
name="dy"
path="/Users/youhan/Documents/FYP/mapping_data/"+name+"/raw/"
if not os.path.exists(path):
    os.makedirs(path)
#60 times per second you can do the math for 17 ms
camera_port = 0
camera = cv2.VideoCapture(camera_port)

#### Calibration part
index=1
for i in range(0,3):
	for j in range(0,3):
		draw.circle(screen, BLACK, previous, radius)
		display.update()
		previous=(720*j+(1-j)*radius, 450*i+(1-i)*radius)
		draw.circle(screen, RED, previous, radius)
		display.update()
		time.sleep(1.1)
		
		return_value, image = camera.read()
		if i or j:
			cv2.imwrite(path+name+"_cali_"+str(index)+".png", image)
			index+=1
		time.sleep(0.5)
		print "cali_"+str(index)+":", previous[0]-720, previous[1]*-1
		  # so that others can use the camera as soon as possible
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				sys.exit()
time.sleep(1)

return_value, image = camera.read()
cv2.imwrite(path+name+"_cali_"+str(index)+".png", image)

### Calibration end


### The second rectangle
index=1
for i in range(0,2):
	for j in range(0,2):
		draw.circle(screen, BLACK, previous, radius)
		display.update()
		previous=(720*j+360, 450*i+225)
		draw.circle(screen, RED, previous, radius)
		display.update()
		time.sleep(1.1)
		
		return_value, image = camera.read()
		if i or j:
			cv2.imwrite(path+name+"_ma1_"+str(index)+".png", image)
			index+=1
		time.sleep(0.5)
		print "ma1_"+str(index)+":", previous[0]-720, previous[1]*-1
		  # so that others can use the camera as soon as possible
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				sys.exit()
time.sleep(1)

return_value, image = camera.read()
cv2.imwrite(path+name+"_ma1_"+str(index)+".png", image)
### The end of second

### The third rectangle
index=1
for i in range(0,3):
	for j in range(0,2):
		draw.circle(screen, BLACK, previous, radius)
		display.update()
		previous=(720*j+360, 450*i+(1-i)*radius)
		draw.circle(screen, RED, previous, radius)
		display.update()
		time.sleep(1.1)

		return_value, image = camera.read()
		if i or j:
			cv2.imwrite(path+name+"_ma2_"+str(index)+".png", image)
			index+=1
		time.sleep(0.5)
		print "ma2_"+str(index)+":", previous[0]-720, previous[1]*-1
		  # so that others can use the camera as soon as possible
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				sys.exit()
time.sleep(1)

return_value, image = camera.read()
cv2.imwrite(path+name+"_ma2_"+str(index)+".png", image)
### The end of third


### The fourth group
index=1
for i in range(0,2):
	for j in range(0,3):
		draw.circle(screen, BLACK, previous, radius)
		display.update()
		previous=(720*j+(1-j)*radius, 450*i+225)
		draw.circle(screen, RED, previous, radius)
		display.update()
		time.sleep(1.1)
		
		return_value, image = camera.read()
		if i or j:
			cv2.imwrite(path+name+"_ma3_"+str(index)+".png", image)
			index+=1
		time.sleep(0.5)
		print "ma3_"+str(index)+":", previous[0]-720, previous[1]*-1
		  # so that others can use the camera as soon as possible
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				sys.exit()
time.sleep(1)

return_value, image = camera.read()
cv2.imwrite(path+name+"_ma3_"+str(index)+".png", image)
### The end of fourth

###validata
index=1
for i in range(0,3):
	for j in range(0,3):
		draw.circle(screen, BLACK, previous, radius)
		display.update()
		previous=(720*j+(1-j)*radius, 450*i+(1-i)*radius)
		draw.circle(screen, RED, previous, radius)
		display.update()
		time.sleep(1.1)
		
		return_value, image = camera.read()
		if i or j:
			cv2.imwrite(path+name+"_vali_"+str(index)+".png", image)
			index+=1
		time.sleep(0.5)
		print "vali_"+str(index)+":", previous[0]-720, previous[1]*-1
		  # so that others can use the camera as soon as possible
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				sys.exit()
time.sleep(1)

return_value, image = camera.read()
cv2.imwrite(path+name+"_vali_"+str(index)+".png", image)

### end of validate
time.sleep(1)
del(camera)