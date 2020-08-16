import numpy as np
import cv2
import pickle


#
width = 640
height = 480
threshold = 0.66

cap = cv2.VideoCapture(0)

cap.set(3,width)
cap.set(4,height)
#getting the trainred mode using pickle


with open("Training.py", "rb") as f:
 train_data = f.read()

#sending an image to test



while True:
    success,imgOriginal=cap.read()
    img=np.asarray(imgOriginal)
    #img=cv2.resize(img,(32,32))
    #img=preprocess(img)  #run it through preprocess function
    #img=img.reshape(1,32,32,1)

    #Predict

    #classIndex = train_data.predict_classes(img)
    #print(classIndex)

    #predictions=train_data.predict(img)

    #probVal = np.amax(predictions) #gives element with highest prediction
    #print (probVal)

   # if probVal > threshold:
       # cv2.putText(imgOriginal,str(classIndex) +"  "+ str(probVal),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,(0,0,255),1)
       # cv2.imshow("Original Image",imgOriginal)



    #cv2.imshow("Preprocessed Image",img)

    # add a statement for breaking whenever q is pressed
    #if cv2.waitkey(1) & 0xFF==ord('q'):
       #break







