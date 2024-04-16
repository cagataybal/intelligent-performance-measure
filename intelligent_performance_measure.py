# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model

imgsize1=875
imgsize2=656
co=1
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(16, (co, co), input_shape = (imgsize1, imgsize2, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (co, co), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(64, (co, co), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a fourth convolutional layer
classifier.add(Conv2D(128, (co, co), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


training_set = train_datagen.flow_from_directory('dataset/forecast_training',
                                                 target_size = (imgsize1, imgsize2),
                                                 batch_size = 1,
                                                 class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory('dataset/forecast_testing',
                                            target_size = (imgsize1, imgsize2),
                                            batch_size = 1,
                                            class_mode = 'binary')

classifier.fit(training_set,
                         steps_per_epoch=2,
                         epochs = 100,
                         validation_data = test_set,
                         validation_steps=200
                         )

# classifier.save('antimonytestsetcnn.h5')  # creates a HDF5 file 'my_model.h5'


# # Part 3 - Finding the best model

# classifier = load_model('antimonytestsetcnn.h5')   # load trained CNN 


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import zeros
from keras.preprocessing import image
from PIL import Image


test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory('dataset/forecast_testing',
                                            target_size = (imgsize1, imgsize2),
                                            batch_size = 1,
                                            class_mode = 'binary')

len_result=int(len(test_set)/2)+1
id=np.empty(len_result, dtype='f')
result=np.empty(len_result, dtype='f')
for num in range(1,len_result):
    test_image = Image.open('dataset/forecast_testing/realdata/output_forecast-'+str(num)+'.jpg')
    # test_image = image.load_img('dataset/forecast_testing/realdata/output_forecast-2.jpg', target_size = (300, 300))
    # plt.imshow(test_image)
    # test_image = realdata_test_set[num-1][0]
    test_image = test_image.resize([imgsize2,imgsize1])    
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis = 0)
    result[num-1] = classifier.predict(test_image)
    id[num-1]=num
    print(result[num-1])
    
numarator = list(range(0, len_result))
nresult = np.array([result, numarator]).T
nresult = nresult[nresult[:, 0].argsort()]
print(nresult[len_result-1,1]+1,nresult[len_result-1,0])
print(nresult[len_result-2,1]+1,nresult[len_result-2,0])
print(nresult[len_result-3,1]+1,nresult[len_result-3,0])
print(nresult[len_result-4,1]+1,nresult[len_result-4,0])
print(nresult[len_result-5,1]+1,nresult[len_result-5,0])

# result_max = np.where(result == np.amax(result))
# result.sort()
# print(result_max[0]+1)
# print(result[result_max[0]])

######

# classifier2 = load_model('antimonycnn.h5')   # load trained CNN 

# test_datagen2 = ImageDataGenerator(rescale = 1./255)

# test_set2 = test_datagen2.flow_from_directory('dataset/forecast_forecasting',
#                                             target_size = (imgsize1, imgsize2),
#                                             batch_size = 1,
#                                             class_mode = 'binary')

# len_result2=int(len(test_set2)/2)+1
# id2=np.empty(len_result2, dtype='f')
# result2=np.empty(len_result2, dtype='f')
# for num in range(1,len_result2):
#     test_image2 = Image.open('dataset/forecast_forecasting/testdata/output_forecast-'+str(num)+'.jpg')
#     # test_image = image.load_img('dataset/forecast_testing/realdata/output_forecast-2.jpg', target_size = (300, 300))
#     # plt.imshow(test_image)
#     # test_image = realdata_test_set[num-1][0]
#     test_image2 = test_image2.resize([imgsize2,imgsize1])    
#     test_image2 = image.img_to_array(test_image2)
#     test_image2 = test_image2 / 255
#     test_image2 = np.expand_dims(test_image2, axis = 0)
#     result2[num-1] = classifier2.predict(test_image2)
#     id2[num-1]=num
#     print(result2[num-1])
    

# result_max2 = np.where(result2 == np.amax(result2))
# print(result_max2[0]+1)
# print(result2[result_max2[0]])
    
