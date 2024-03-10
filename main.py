# Make the creating of our model a little easier
import pathlib

from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# Plot the validation and training data separately
def plot_loss_curves(history):
    """
  Returns separate loss curves for training and validation metrics.
  """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


def load_and_prep_image(filename: str, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.
    return img


def pred_and_plot(model, filename: str, class_name):
    img = load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))
    pred_class = class_name[int(tf.round(pred))]
    plt.imshow(img)
    plt.title(f"Prediction {pred_class}")
    plt.axis(False)
    plt.show()


print(tf.__version__)

# Set up the train and test directories
train_dir = "pizza_steak/train/"
test_dir = "pizza_steak/test/"
data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
tf.random.set_seed(42)
# Create ImageDataGenerator training instance with data augmentation
train_datagen_augmented = ImageDataGenerator(rescale=1 / 255.,
                                             rotation_range=20,  # rotate the image slightly between 0 and 20 degrees
                                             # (note: this is an int not a float)
                                             shear_range=0.2,  # shear the image
                                             zoom_range=0.2,  # zoom into the image
                                             width_shift_range=0.2,  # shift the image width ways
                                             height_shift_range=0.2,  # shift the image height ways
                                             horizontal_flip=True)  # flip the image on the horizontal axis

# Create ImageDataGenerator training instance without data augmentation
train_datagen = ImageDataGenerator(rescale=1 / 255.)

# Create ImageDataGenerator test instance without data augmentation
test_datagen = ImageDataGenerator(rescale=1 / 255.)

# Import data and augment it from training directory
print("Augmented training images:")
train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,target_size=(224, 224),  batch_size=32,
                                                                   class_mode='binary', shuffle=True)

# Create non-augmented data batches
print("Non-augmented training images:")
train_data = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='binary',
                                               shuffle=True)  # Don't shuffle for demonstration purposes

print("Unchanged test images:")
test_data = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='binary')

# Create the model
model = Sequential([
    Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
    MaxPool2D(),
    Conv2D(10, 3, activation='relu'),
    MaxPool2D(),
    Conv2D(10, 3, activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Model summary
model.summary()

# Fit the model
history = model.fit(train_data_augmented,  # now the augmented data is shuffled
                    epochs=5, steps_per_epoch=len(train_data_augmented), validation_data=test_data,
                    validation_steps=len(test_data))

steak = load_and_prep_image("03-steak.jpeg")
pred = model.predict(tf.expand_dims(steak, axis=0))
pred_and_plot(model, "03-steak.jpeg", class_names)
plot_loss_curves(history)
