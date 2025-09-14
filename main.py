import tensorflow as tf
import os
import numpy as np

# setting fashion_mnist as the dataset to be using
data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0               # dividing by 255 so that every greyscale value is between 0-1

checkpoint_path = "training_1/test.weights.h5" #location that weights will be stored in
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)
#setting model checkpoint to the path specified
# saving only the weights, no other data is needed.

def create_model(): #creating layers of a neural network
    temp_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)), # takes 2d array and flattens it into 1d array
        tf.keras.layers.Dense(30, activation=tf.nn.sigmoid), #
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    temp_model.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return temp_model


model = create_model()
model.load_weights(checkpoint_path)


"""
model.fit(training_images, training_labels, epochs=5, callbacks = [cp_callback]) #callback argument automatically saves the file
model.evaluate(test_images, test_labels) #uncomment to train the model
"""


class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]




def predict_new_image(image_path, model):
    """Loads a new image from disk and predicts its class using the trained model."""
    try:
        # Load the image and decode it
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=1, expand_animations=False)

        # Resize and cast to float32
        img = tf.image.resize(img, (28, 28))
        img = tf.cast(img, tf.float32)

        # Normalize the pixel values (0-255 -> 0-1)
        img = img / 255.0

        # Reshape the image to match the model's input shape (1, 28, 28)
        # This explicitly defines the shape, which should solve the error.
        img = tf.reshape(img, [1, 28, 28])

        # Make a prediction using the trained model
        predictions = model.predict(img)

        # Find the index of the highest prediction value (the predicted class)
        return np.argmax(predictions)

    except tf.errors.NotFoundError:
        print(f"Error: The file at {image_path} was not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")
# --- Example Usage ---
# Replace 'path/to/your/image.jpg' with the actual path to your image file
my_image_path = "IMG_4225.jpg"
print(class_names[predict_new_image(my_image_path, model)])