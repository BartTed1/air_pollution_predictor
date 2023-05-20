import tensorflow as tf

class NeuralNetwork:
    def __init__(self, data, labels):
        self.model = None
        self.input_dim = 5  # number of features
        self.output_dim = 1  # number of output labels
        
        # Define the data used to train the model
        self.labels = labels
        self.data = data

    def build_and_train(self):
        # Define the layers of the neural network
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.input_dim,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='linear'),
            tf.keras.layers.Dense(self.output_dim)
        ])

        # Compile the model with an appropriate loss function and optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(loss='mse', optimizer=optimizer)
        
        # Train the model on the provided data-label set
        self.model.fit(self.data, self.labels, epochs=100, verbose=1)
        return self.model

    def predict(self):
        # Use the trained model to make predictions on new data
        new_data = [[3, 19, 0.2434, 5.41, 5.71]]
        predictions = self.model.predict(new_data)
        print(predictions)
        return predictions
    
    def load(self, path):
        # Load the model
        self.model = tf.keras.models.load_model(path)
        return self.model

    def save(self):
        # Save the model
        self.model.save('model.h5')
        return 1