# importing libraries
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Conv2D
from keras.preprocessing.image import ImageDataGenerator


# initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Full Connected Layer
classifier.add(Dense(output_dim=128, activation='relu'))
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

classifier.fit_generator(
    training_set,
    steps_per_epoch=8000,
    epochs=25,
    validation_data=test_set,
    validation_steps=2000
)
