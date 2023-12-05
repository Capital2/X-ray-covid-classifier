from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing import image
import numpy as np


img_path = '/your/test/image/image.png' 
img = image.load_img(img_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

loaded_model = load_model("covid_pneumonia_classification_model.h5")
predictions = loaded_model.predict(img_array)
class_labels = ['COVID', 'NORMAL', 'PNEUMONIA']  

predicted_class = class_labels[np.argmax(predictions)]
confidence = np.max(predictions)

print(f'Predicted Class: {predicted_class}')
print(f'Confidence: {confidence:.2%}')