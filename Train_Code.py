
from simple_unet_model import simple_unet_model   #Use normal unet model
from keras.utils import normalize
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


image_directory = 'path/1600idark/'
mask_directory = 'path/1600mdark/'


#SIZE = 256
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'tif'):
        #print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name, 0)
        image = Image.fromarray(image)
 #       image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'tif'):
        image = cv2.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
  #      image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))

#Normalize images
image_dataset = np.expand_dims((np.array(image_dataset)),3) /255.
#D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.75, random_state = 42)

#Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()


###############################################################
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()


#If starting with pre-trained weights. 
#model.load_weights('solar1.hdf5')

from keras.models import load_model
model = load_model('solar12.hdf5')
history = model.fit(X_train, y_train, 
                    batch_size = 10, 
                    verbose=1,
                    epochs=20,                   
                    validation_data=(X_test, y_test), 
                    shuffle=False)


#Add Callbacks
from keras.callbacks import ModelCheckpoint 
filepath="weights/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5" #File name includes epoch and validation accuracy.
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


history = model.fit(X_train, y_train, 
                    batch_size = 10, 
                    verbose=1, 
                    epochs=20, 
                    validation_data=(X_test, y_test), 
                    shuffle=False, callbacks = [checkpoint])

model.save('solardark.hdf5')

#Add Callbacks, e.g. ModelCheckpoints, earlystopping, csvlogger.
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

#ModelCheckpoint callback saves a model at some interval. 
filepath="solar2.hdf5" #File name includes epoch and validation accuracy.
#Use Mode = max for accuracy and min for loss. 
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
############################################################
#Evaluate the model


	# evaluate model
_, accuracy = model.evaluate(X_test, y_test)
print("Accuracy = ", (accuracy * 100.0), "%")


#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#acc = history.history['acc']
accuracy = history.history['accuracy']
#val_acc = history.history['val_acc']
val_accuracy = history.history['val_accuracy']

plt.plot(epochs, accuracy, 'y', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
##################################
#IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.2

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

#######################################################################
#Predict on a few images
model = get_model()
model.load_weights('solar12.hdf5') #Trained for 50 epochs and then additional 100
#model.load_weights('mitochondria_gpu_tf1.4.hdf5')  #Trained for 50 epochs

test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8)


test_img_other_norm =[]

test_img_other = cv2.imread('test2.tif', 0) 
#test_img_other = cv2.imread('test7.png', 0)
test_img_other = Image.fromarray(test_img_other)
test_img_other_norm.append(np.array(test_img_other))


test_img_other_norm = np.expand_dims((np.array(test_img_other)),2)/255.
test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
test_img_other_input=np.expand_dims(test_img_other_norm, 0)

#Predict and threshold for values above 0.5 probability
#Change the probability threshold to low value (e.g. 0.05) for watershed demo.
prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.2).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Input Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Ground Truth')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on Input Image')
plt.imshow(prediction, cmap='gray')
plt.subplot(234)
plt.title('Test Image')
plt.imshow(test_img_other, cmap='gray')
plt.subplot(235)
plt.title('Prediction on Test Image')
plt.imshow(prediction_other, cmap='gray')
plt.show()

#plt.imsave('test/Test3.png', test_img_other, cmap='gray')
#plt.imsave('output/OUTPUT40.png', prediction_other, cmap='gray')

patch_size=256
def prediction(model, image, patch_size):
    segm_img = np.zeros(image.shape[:2])  #Array with zeros to be filled with segmented values
    patch_num=1
    for i in range(0, image.shape[0], 256):   #Steps of 256
        for j in range(0, image.shape[1], 256):  #Steps of 256
            #print(i, j)
            single_patch = image[i:i+patch_size, j:j+patch_size]
            single_patch_norm = np.expand_dims((np.array(single_patch)), 2) /255.
            single_patch_shape = single_patch_norm.shape[:2]
            single_patch_input = np.expand_dims(single_patch_norm, 0)
            single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.2).astype(np.uint8)
            segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])
          
            print("Finished processing patch number ", patch_num, " at position ", i,j)
            patch_num+=1
    return segm_img

##########
#Load model and predict

#Large image
large_image = cv2.imread('test1.png', 0)
segmented_image = prediction(model, large_image, patch_size)
plt.hist(segmented_image.flatten())  #Threshold everything above 0

plt.imsave('Output_Dark/kkk.jpg', segmented_image, cmap='gray')

plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.title('Large Image')
plt.imshow(large_image, cmap='gray')
plt.subplot(222)
plt.title('Prediction of large Image')
plt.imshow(segmented_image, cmap='gray')
plt.show()


##################################
#Watershed to convert semantic to instance
#########################
from skimage import measure, color, io

#Watershed
img = cv2.imread('output2.jpg')  #Read as color (3 channels)
img_grey = img[:,:,0]

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(img_grey,cv2.MORPH_OPEN,kernel, iterations = 2)

sure_bg = cv2.dilate(opening,kernel,iterations=10)
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

ret2, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(),255,0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

ret3, markers = cv2.connectedComponents(sure_fg)
markers = markers+10

markers[unknown==255] = 0

markers = cv2.watershed(img, markers)
img[markers == -1] = [0,255,255]  

img2 = color.label2rgb(markers, bg_label=0)

cv2.imshow('Overlay on original image', large_image)
#plt.imsave('kk.jpg', large_image)
cv2.imshow('Colored Grains', img2)
#plt.imsave('colour1.jpg', img2)
cv2.waitKey(0)


props = measure.regionprops_table(markers, intensity_image=img_grey, 
                              properties=['label',
                                          'area', 'equivalent_diameter',
                                          'mean_intensity', 'solidity'])
    
import pandas as pd
df = pd.DataFrame(props)
df = df[df.mean_intensity > 100]  #Remove background or other regions that may be counted as objects
   
print(df.head())
