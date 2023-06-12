import tensorflow as tf
from keras.callbacks import ModelCheckpoint

class NeuralNetwork:
    def __init__(self, *args):
        """
        :param args: either a path to a saved model or a data-label set
        """
        if len(args) == 0:
            raise Exception("No data provided")
        elif len(args) == 1 and type(args[0]) == str:
            self.model = self.load(args[0])
        elif len(args) == 2 and type(args[0]) == list and type(args[1]) == list:
            self.data = args[0]
            self.labels = args[1]

        self.input_dim = 5  # number of features
        self.output_dim = 1  # number of output labels
        
        # Define the data used to train the model

    def build_and_train(self):
        # Define the layers of the neural network
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.input_dim,)),
            #tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='linear'),
            tf.keras.layers.Dense(self.output_dim)
        ])

        # Define the checkpoint callback to save the best model based on validation loss
        checkpoint_callback = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

        # Compile the model with an appropriate loss function and optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(loss='mse', optimizer=optimizer)
        
        # Train the model on the provided data-label set
        # TODO Use x% of the data for validation
        self.model.fit(self.data,
                       self.labels,
                       # validation_data=(self.data, self.labels),
                       epochs=100,
                       verbose=1,
                       callbacks=[checkpoint_callback]
                        )

        # TODO Load the best model based on the saved checkpoint
        # self.model = tf.keras.models.load_model('best_model.h5')
        return self.model

    def predict(self, X_data):
        # Use the trained model to make predictions on new data
        predictions = self.model.predict(X_data, verbose=0)
        return predictions.tolist()[0][0]
    
    def load(self, path):
        # Load the model
        self.model = tf.keras.models.load_model(path)
        return self.model

    def save(self):
        # Save the model
        self.model.save('model.h5')
        return 1
