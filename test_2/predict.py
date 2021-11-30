from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2

model = load_model('multiclass_model80_77.h5')
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def predict(path:str):
    image=np.expand_dims(cv2.imread(path)*(1.0/255.0), axis=0)
    predictions=model.predict(image)
    print(predictions)
    
    
predict('./img2.jpg')