import tensorflow as tf
from CustomModelCheckPoint import CustomModelCheckpoint
from tensorflow.keras import datasets, layers, models
from keras.optimizers import SGD

import matplotlib.pyplot as plt
class CNNModel(object):
    def create_model(self, learning_rate):
        # highest accuracy liveness on non-diffused iages- 16 (13,13) => 32 (7,7) => 64 (5,5) => 64 => 1 - acc = 94.2
        model = models.Sequential()
        model.add(layers.Conv2D(16, (15, 15), activation='relu', input_shape=(64, 64,1)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (7, 7), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        #model.add(layers.Dense(2))  # for sparse_categorial_crossentropy - then choose 2 neurons in next layer
        model.add(layers.Dense(1, activation='sigmoid'))
        opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=1e-6)
        model.compile(optimizer= opt , loss= tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
        return model

    def load_model(self,model_file_name):
        model = tf.keras.models.load_model(model_file_name)
        return model

    def train_model(self, model, train_images,train_labels,test_images, test_labels, epochs, best_accuracy,file_to_save, name):
     
        cbk = CustomModelCheckpoint(best_accuracy,file_to_save, name)  # so that we can save the best model
        history = model.fit(train_images, train_labels, epochs=epochs, callbacks=[cbk], 
                        validation_data=(test_images, test_labels))
        #plt.plot(history.history['accuracy'], label='accuracy')
        #plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        #plt.xlabel('Epoch')
        #plt.ylabel('Accuracy')
        #plt.ylim([0.5, 1])
        #plt.legend(loc='lower right')
        #plt.show()
        return model

    def evaluate(self, model, test_images, test_labels):
        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
        return test_loss, test_acc