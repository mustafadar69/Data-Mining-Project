import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

from zipfile import ZipFile

from tensorflow.python.keras.callbacks import LearningRateScheduler

file_name="myData.zip"

with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print('done')


#making list of all the folders in the project...retrieveing their names and infor
path = 'myData'


#Parameters

testRatio = 0.2
ValidationRatio = 0.2
images = []
classNum=[]#save the corresponding id of each class
Mylist = os.listdir(path)
imageDimension=(32,32,3)
##########################
#print((Mylist))
x=len(Mylist)-1
print("Total number of classes Detected = ",x)
print(Mylist)

NoOfclasses=x
#loop for getting the images from all the folders
print("Importing Classes")
for x in range (0,NoOfclasses):
    piclist = os.listdir(path + "/"+str(x))
    for y in piclist:
        currentImg = cv2.imread(path + "/"+str(x)+"/"+y)
        #currentImg = cv2.resize(currentImg(32,32))#resizing to 32 but because 180 will be expensive
        images.append(currentImg)
        classNum.append(x)#corresponding class list in which all ids are stored

    print(x, end=" ")
print(" ")

#tells how much pictures have been imported
print("Total number of images imported = ",len(images))
#print(len(classNum))


#convert into numpy array
images=np.array(images)
classNum=np.array(classNum)

print(images.shape)
print(classNum.shape)

#Splitting data for training and testing
#splitting in the ratio of 20 and 80 using sklearn

x_train, x_test, y_train, y_test = train_test_split(images, classNum, test_size = testRatio)
x_train,x_validation,y_train,y_validation= train_test_split(x_train,y_train,test_size = ValidationRatio)


print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

#telling total no of images of 0
numberOfSamples=[]
for z in range(0,NoOfclasses):

    #print(len(np.where(y_train==z)[0]))
    numberOfSamples.append(len(np.where(y_train==z)[0]))
print(numberOfSamples)


plt.figure(figsize = (10,5))
plt.bar(range(0,NoOfclasses),numberOfSamples)
plt.title("Number of Images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()


#Pre processing of images
#makers the lighting of image distributed equally
def preprocess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # makers the lighting of image distributed equally
    img = cv2.equalizeHist(img)
    #normalizing values and restrict it to 0 and 1 and it's better for training
    img = img/255
    return img

#picking image form x train and its index 30
img=preprocess(x_train[30])
#resizing so we can see it clearly
img=cv2.resize(img,(300,300))
cv2.imshow("Pre-Processed",img)
cv2.waitKey(0)

#run a function over array of elements..it takes x-train input and pre process it and then again convert it into np array and again store it into the x-train
x_train = np.array(list(map(preprocess,x_train)))
x_test = np.array(list(map(preprocess,x_test)))
x_validation = np.array(list(map(preprocess,x_validation)))

#add a depth of 1 to let work CNN perfectly

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)

#keras library is used for augmented image
#creating augmented images
#10% is considered and 0.1 in width shift range and height shift range
#shear range is the magnetude
#rotation would be in degrees

datagen=ImageDataGenerator(width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.2,shear_range=0.1,
                           rotation_range=10)
#generating as we go along the training process
datagen.fit(x_train)

#encode our matices

y_train = to_categorical(y_train,NoOfclasses)
y_test = to_categorical(y_test,NoOfclasses)
y_validation = to_categorical(y_validation,NoOfclasses)

#creating model

def Model():
    numFilters = 60
    sizeFilter = (5,5)
    sizeFilter2 = (3,3)
    sizePool = (2,2)
    noOfNode = 500

    model = Sequential()
    # creating the first layer and adding convolutional layer,consists of height and image usually smaller then original

    model.add((Conv2D(numFilters,sizeFilter,input_shape=(imageDimension[0],imageDimension[1],1),activation='relu')))

    model.add((Conv2D(numFilters,sizeFilter,activation='relu')))
    #maxpool is used to reduce dimensions of image or hidden layer amtrix to allow assumptions to be accurate
    model.add(MaxPooling2D(pool_size=sizePool))

    model.add((Conv2D (numFilters/2 , sizeFilter2 , activation = 'relu')))
    model.add((Conv2D (numFilters/2 , sizeFilter2 , activation = 'relu')))
    model.add(MaxPooling2D (pool_size = sizePool))
   #helping to reduce overfitting and making it generic
    model.add(Dropout(0.5))

    #reshape the input data into dormat suitable for CNN using X Train
    model.add(Flatten())
    #Dense(10) has 10 neurons.Takes output from prev layer neurons and providing output to next neuron
    model.add(Dense(noOfNode,activation='relu'))
    model.add(Dense(NoOfclasses,activation='softmax'))

    model.compile(Adam(lr=0.001),loss='categorical crossentropy',metrics=['accuracy'])
    return model


model = Model()
print(model.summary())

#define the optimizer

optimizer=Adam(lr=1e-4)

# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

#run the training
batchvalue=20
epochsVal=10
stepPerEpoch=2000

#history = model.fit_generator(datagen.flow(x_train,y_train,batch_size =batchvalue ),
 #                             validation_data = (x_validation,y_validation),shuffle = 1)

# Without data augmentation i obtained an accuracy of 0.98114
history = model.fit(x_train, y_train, batch_size = batchvalue, epochs = epochsVal,
           validation_data = (x_validation, y_validation), verbose = 2)

datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1)  # randomly shift images vertically (fraction of total height)

datagen.fit(x_train)



annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)


# Fit the model
history = model.fit(datagen.flow(x_train,y_train, batch_size=batchvalue),
                    epochs = epochsVal, validation_data = (x_validation,y_validation),
                    verbose = 0, steps_per_epoch=x_train.shape[0]
                                                 // batchvalue,callbacks=[annealer])



#plotting our training model

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['value_loss'])
plt.legend(['training','Validation'])
plt.title("Loss")
plt.xlabel('Epoch')


#plotting accuracy

plt.figure(2)
plt.plot(history.history['Accuracy'])
plt.plot(history.history['value_Accuracy'])
plt.legend(['training','Validation'])
plt.title("Accuracy")
plt.xlabel('Epoch')
plt.show()
score = model.evaluate(x_test,y_test,verbose=0)

print("Test Score = ",score[0])
print("Accuracy = ", score[1])


#stroing model as pickle obj for webcam

pickle_x=open("Training.py","wb")
#pickle.dump(model,pickle_x)
pickle_x.close()


