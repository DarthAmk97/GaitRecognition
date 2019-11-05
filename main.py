from tensorflow.keras.layers import (
    Convolution2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    ConvLSTM2D,
    LSTM,
    CuDNNLSTM,
    TimeDistributed
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
import simplejson as sj
from keras.datasets import cifar10

def create_model():
    model = Sequential()
    model.add((Convolution2D(64, (3, 3), input_shape=(200, 300, 3), activation='relu')))
    model.add((MaxPooling2D(pool_size=(2, 2))))
    #model.add(LSTM(units=32,input_shape=(200, 300, 3, None)))
    model.add(Convolution2D(32, (3, 3), input_shape=(200, 300, 3), activation='relu',name='conv1.1'))
    model.add(Convolution2D(16, (3, 3), input_shape=(200, 300, 3), activation='relu',name='conv1.2'))
    model.add(Convolution2D(256, (3, 3), input_shape=(200, 300, 3), activation='relu',name='conv1.3'))
    model.add((MaxPooling2D(pool_size=(2, 2))))
    model.add((Flatten()))
    model.add((BatchNormalization()))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=4, activation='softmax'))
    return model

def train(model, training_set, test_set):
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(
            training_set,
            steps_per_epoch=None,
            epochs=50,
            verbose=1,
            validation_data=test_set,
            validation_steps=None
    )

def save_model(model):
    print("Saving...")
    model.save_weights("model.h5")
    print(" [*] Weights")
    open("model.json", "w").write(
            sj.dumps(sj.loads(model.to_json()), indent=4)
    )
    print(" [*] Model")

def load_model():
    print("Loading...")
    json_file = open("model.json", "r")
    model = model_from_json(json_file.read())
    print(" [*] Model")
    model.load_weights("model.h5")
    print(" [*] Weights")
    json_file.close()
    return model

def dataset_provider(datagen):
    return datagen.flow_from_directory(
        './Images',
        target_size=(200, 300),
        batch_size=16,
        class_mode='categorical'
    )

# Primary datagen
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
# Validation datagen
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Primary Set for training
training_set = dataset_provider(train_datagen)
# Secondary / Test set for validation
test_set = dataset_provider(test_datagen)


config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.Session(config=config)
K.set_session(sess)
#img_load=

#model = create_model()
#train(model,training_set,test_set)
#save_model(model)
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(training_set, test_set, test_size=0.3, random_state=0)

from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
weights=loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.summary()
# evaluate loaded model on test data
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
score = loaded_model.evaluate(test_set, weights)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

#import matplotlib.pyplot as plt
#plt.plot(training_set, test_set)
#plt.xlabel('Training')
#plt.ylabel('Test')
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

#y_pred1 = loaded_model.predict(training_set)
#y_pred = np.argmax(y_pred1,axis=1)
#print(y_pred)
#print(y_pred.shape)
#print(X_train)
#print(X_test)
# Print f1, precision, and recall scores
#print(precision_score(y_test, y_pred , average="macro"))
#print(recall_score(y_test, y_pred , average="macro"))
#print(f1_score(y_test, y_pred , average="macro"))
import matplotlib as plt
plt.plot(training_set, test_set)
plt.xlabel('training')
plt.ylabel('test')
