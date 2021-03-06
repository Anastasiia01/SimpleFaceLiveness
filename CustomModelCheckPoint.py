import tensorflow as tf
import json

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, best_accuracy, model_file, model_name):
        # set the lastvalacc to a higher value (best that was found) in the 
        # previous training so that it looks for a better model than the last one
        self.lastvalacc = best_accuracy  #0.9796 #orig inception v4 0.9788, # 0.9703 Inception v4 diffused # last best
        self.path = "config.json"
        self.model_file = model_file
        self.model_name = model_name
    
    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        
        print(f"epoch: {epoch}, train_acc: {logs['accuracy']}, valid_acc: {logs['val_accuracy']}")

        if logs['val_accuracy'] > self.lastvalacc: # your custom condition
            self.model.save(self.model_file, overwrite=True)  # or model_Inception_v1.h5
            self.lastvalacc = logs['val_accuracy']
            with open(self.path) as file:
                print("Reading from json ... ")
                data = json.load(file)
            data[self.model_name]['accuracy'] = self.lastvalacc
            with open(self.path,'w') as file:
                print(f"Updating accuracy to be {self.lastvalacc} ... ")
                json.dump(data, file, indent=2)
            print(f'--------better model on epoch found with accuracy = {self.lastvalacc} ========$$$$$$$$$$')