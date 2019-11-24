from tensorflow.keras.layers import (
    Convolution2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    ConvLSTM2D,
    Conv3D,
    LSTM,
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
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import numpy as np
from VidToFrame import VideoToFrames
import os

def create_model(): #main one
    model = Sequential()
    model.add((Convolution2D(64, (3, 3), input_shape=(352, 240, 3), activation='relu')))
    model.add((MaxPooling2D(pool_size=(2, 2))))
    model.add((Convolution2D(128, (3, 3), activation='relu')))
    model.add((MaxPooling2D(pool_size=(2, 2))))
    model.add((Flatten()))
    model.add((BatchNormalization()))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=3, activation='softmax'))
    return model


def train(model,train_datagen,test_datagen):
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(
            train_generator,
            steps_per_epoch=None,
            epochs=50,
            verbose=1,
            validation_data=validation_generator,
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

def EvaluateModel():
# load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
# load weights into new model
    weights=loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
# evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    score = loaded_model.evaluate(validation_generator, weights)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

def RecognizeOnDirectory():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    weights = loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    folder_path = "D:\GaitRecognition2\gait\core\\00_3"
    import os
    from keras.preprocessing import image
    images = []
    for img in os.listdir(folder_path):
        img = os.path.join(folder_path, img)
        img = image.load_img(img, target_size=(352, 240))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)

    count = 0
    images = np.vstack(images)
    classes = loaded_model.predict_classes(images)
    print(classes)
    testy = np.asarray(classes)
    print(len(classes))
    print("class 0, 1, 2 values: greater is predicted person: ")
    wow1 = np.count_nonzero(classes == 0)
    print(wow1)
    wow2 = np.count_nonzero(classes == 1)
    print(wow2)
    wow3 = np.count_nonzero(classes == 2)
    print(wow3)
    predicted_class_indices = np.argmax(classes, axis=0)
    print("Class is: ")
    print(predicted_class_indices)

def ActivateGPU():
    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
    sess = tf.Session(config=config)
    config.gpu_options.allow_growth = True
    K.set_session(sess)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    # this is the augmentation configuration we will use for testing:
    # only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
train_generator = train_datagen.flow_from_directory(
            './Images',  # this is the target directory
            target_size=(352, 240),  # all images will be resized to 150x150
            batch_size=40,
            class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels
    # this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
            './test',
            target_size=(352, 240),
            batch_size=40,
            class_mode='categorical')

def main():

    model = create_model()
    train(model,train_datagen,test_datagen)
    save_model(model)
    EvaluateModel()
    RecognizeOnDirectory()

    #call when input is video
    #obj = VideoToFrames()
    #obj.run()

if __name__ == "__main__":
        main()
