import tensorflow as tf
class CNN:
    def __init__(self):
        self.model = tf.keras.models.Sequential()
        self.modeltrained = False
        self.modelbuilt = False
    def build_and_compile_model(self):
        if self.modelbuilt:
            return
        #Convolutional layer
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
        #Max pooling layer
        self.model.add(tf.keras.layers.MaxPool2D())
        #Flattened layer
        self.model.add(tf.keras.layers.Flatten())
        #Hidden layer
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        #Dropout layer
        self.model.add(tf.keras.layers.Dropout(0.2))
        #Output layer
        self.model.add(tf.keras.layers.Dense(10, activation='softmax'))
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.modelbuilt = True

    def train_and_evaluate_model(self):
        if not self.modelbuilt:
            raise Exception("Build and train the model first!")
        if self.modeltrained:
            return
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_train, x_test = x_train / 255.0, x_test / 255.0
        self.model.fit(x=x_train, y=y_train, epochs=5)
        test_loss, test_acc = self.model.evaluate(x=x_test, y=y_test)
        print('\nTest accuracy:', test_acc)
        self.modeltrained = True

    def save_model(self):
        if not self.modelbuilt:
            raise Exception("Build and compile the model first!")
        if not self.modeltrained:
            raise Exception("Train and evaluate the model first!")
        self.model.save("cnn.hdf5", overwrite=True)
