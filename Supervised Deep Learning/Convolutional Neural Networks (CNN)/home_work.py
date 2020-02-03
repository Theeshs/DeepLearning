# importing libraries
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Conv2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.callbacks import Callback


class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''

    def on_epoch_end(self, epoch, logs={}):
        # self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n" \
        #     .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1

    def on_train_end(self, logs={}):
        self.losses += "Training begins...\n"


# initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
# Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# adding dropout
# classifier.add(Dropout(0.1))

# More Layers
classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.5))
classifier.add(Convolution2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.2))

# Step 3 - Flattening
classifier.add(Flatten())

# Full Connected Layer
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# Completing the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Image prepossessing
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    r"C:\Users\ChampWk38\Desktop\Deep_Learning_A_Z\Volume 1 - Supervised Deep Learning\Part 2 - Convolutional Neural Networks (CNN)\Section 8 - Building a CNN\Convolutional_Neural_Networks\dataset\training_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    r"C:\Users\ChampWk38\Desktop\Deep_Learning_A_Z\Volume 1 - Supervised Deep Learning\Part 2 - Convolutional Neural Networks (CNN)\Section 8 - Building a CNN\Convolutional_Neural_Networks\dataset\test_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Create a loss history
history = LossHistory()

classifier.fit_generator(
    training_set,
    steps_per_epoch=8000 / 32,
    epochs=90,
    validation_data=test_set,
    validation_steps=2000 / 32,
    workers=12,
    max_queue_size=100,
    callbacks=[history],
    verbose=2
)

test_img_1 = image.load_img(
    r'C:\Users\ChampWk38\Desktop\Deep_Learning_A_Z\Volume 1 - Supervised Deep Learning\Part 2 - Convolutional Neural Networks (CNN)\Section 8 - Building a CNN\Convolutional_Neural_Networks\dataset\single_prediction\cat_or_dog_1.jpg',
    target_size=(64, 64))
test_img_2 = image.load_img(
    r'C:\Users\ChampWk38\Desktop\Deep_Learning_A_Z\Volume 1 - Supervised Deep Learning\Part 2 - Convolutional Neural Networks (CNN)\Section 8 - Building a CNN\Convolutional_Neural_Networks\dataset\single_prediction\cat_or_dog_2.jpg',
    target_size=(64, 64))

test_img_1 = image.img_to_array(test_img_1)
test_img_2 = image.img_to_array(test_img_2)

test_img_1 = np.expand_dims(test_img_1, axis=0)
test_img_2 = np.expand_dims(test_img_2, axis=0)

x = classifier.predict(test_img_1)
print(x)
y = classifier.predict(test_img_2)
print(y)
print(training_set.class_indices)
