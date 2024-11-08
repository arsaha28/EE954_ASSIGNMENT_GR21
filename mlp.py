import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchsummary import summary
import pandas as pd

dataset_location = root = './data/'
training_dataset = datasets.FashionMNIST(dataset_location,
                               train=True,
                               transform=transforms.ToTensor(),
                                 download=True)

test_dataset = datasets.FashionMNIST(dataset_location,
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)
print("training dataset length =", len(training_dataset))
print("test dataset length =", len(test_dataset))

combined_dataset = torch.utils.data.ConcatDataset([training_dataset, test_dataset])
print("combined dataset length =", len(combined_dataset))

#Initializing the ratios for the test, training and validation datasets
train_dataset_ratio = 0.9
validation_dataset_ratio = 0.1
test_dataset_ratio = 1.0 # This is applied to the full test data set

#Initalizing the new values of the training, testing and validation data sizes
train_dataset_size = int(train_dataset_ratio * len(training_dataset))
test_dataset_size = int(test_dataset_ratio * len(test_dataset))
validation_dataset_size = int(validation_dataset_ratio * len(training_dataset))

print("training dataset length =", train_dataset_size)
print("test dataset length =", test_dataset_size)
print("validation dataset length =", validation_dataset_size)
#create the datasets with the sizes

new_train_dataset, new_validation_dataset = torch.utils.data.random_split(training_dataset, [train_dataset_size, validation_dataset_size])
new_test_dataset = torch.utils.data.random_split(test_dataset, [test_dataset_size]) # This is a redundant step but will be useful if the ratios change
print(train_dataset_size)
print(test_dataset_size)
print(validation_dataset_size)
print(combined_dataset.cumulative_sizes)
print(len(new_train_dataset))
print(len(new_test_dataset))
print(len(new_validation_dataset))

sample_image, sample_label = new_train_dataset[0]
print("Min pixel value:", sample_image.min().item())
print("Max pixel value:", sample_image.max().item())


#Defining class names for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Displaying a 4x4 grid of the first 16 images in the dataset with pixel size and pixel range
plt.figure(figsize=(8, 8))  # Set the size of the figure
for i in range(12):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image, label = new_train_dataset[i]

    min_pixel = image.min().item()
    max_pixel = image.max().item()

    plt.imshow(image.reshape((28,28)).squeeze())
    #plt.imshow(image.squeeze())

    # Display the class name, pixel size, and pixel range in the title
    plt.title(f"{class_names[label]}\n28x28 pixels\nRange: {min_pixel:.2f}-{max_pixel:.2f}")

plt.tight_layout()
plt.subplots_adjust(hspace=0.6)
plt.show()  # Show the 6x6 grid

# Now, display a random 15x15 grid of images
W_grid = 10
L_grid = 10

fig, axes = plt.subplots(L_grid, W_grid, figsize = (15,15))
axes = axes.ravel() # Flatten the grid to make it easier to access each subplot, 2D to 1D
n_train = len(new_train_dataset)

for i in np.arange(0, W_grid * L_grid):
    index = np.random.randint(0, n_train)
    sample_image, sample_label = new_train_dataset[index]
    axes[i].imshow(sample_image.reshape((28,28)))
    axes[i].set_title(class_names[sample_label], fontsize = 9)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.3)
# Relu activation function
def relu(x):
    return np.maximum(0, x)

# Derivative of Relu activation function
def relu_derivative(x):
    return np.where(x > 0, 1, 0)
#Softmax function
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

input_layers_config = [
    {'neurons': 36, 'activation': 'relu'},
    {'neurons': 36, 'activation': 'relu'}
]
output_layers_config = [
    {'neurons': 10, 'activation': 'softmax'}
]

class CustomModel(nn.Module):
     def __init__(self):
       super(CustomModel, self).__init__()
       self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            )
     def forward(self, x):
            x = self.conv_layer(x)
            return x
model = CustomModel()
# Create a sample input tensor with shape (1, 1, 28, 28)
input_data = torch.randn(1, 1, 28, 28)
# Pass the input through the model#
cnn_output = model(input_data)
summary(model, input_size=(1, 28, 28))
# Print the output shape
print("Output shape:", cnn_output.shape)
#Fully connected layer
class NeuralNetwork:
    def flatten(self, X):
      batch_size = X.shape[0]
      #Flatten each image/sample to a 1D vector
      return X.reshape(batch_size, -1) # output(batch size, flattened size)


    def __init__(self):
      self.layers = []
      self.activation_functions = []
      self.layer_data = []

    def addnetwork(self, input_size, input_layers_config, output_layers_config):

      """
        - Add the first fully connected layer with the same shape as the flattened CNN output.
        - input_size (int): The size of the input layer.\n",
        - input_layers_config (list): A list of dictionaries specifying the configuration of input layers,
        - output_layers_config (list): A list of dictionaries specifying the configuration of output layers.
        - layers (list): A list to store the layers of the neural network.\n",
        - activation_functions (list): A list to store the activation functions for each layer.\n",

        Args:
        - input_size (int): The size of the input layer.
        - input_layers_config (list): A list of dictionaries specifying the configuration of input layers.
        - output_layers_config (list): A list of dictionaries specifying the configuration of output layers.

        Returns:
        - None

      """
      #addition the first layer
      neurons = input_size  # Same number of neurons as input size
      activation = 'relu'
      new_input_layers_config = []

      # Initialize weights and biases for the first layer
      weights = np.random.randn(input_size, neurons)
      bias = np.zeros((1, neurons))

      # Store layer information
      self.layer_data.append({
          "Layer": "First Layer",
          "Input Neurons": input_size,
          "Output Neurons": neurons,
          "Weights": weights.size,
          "Biases": bias.size,
          "Total Parameters": weights.size + bias.size
      })

      # Append weights, biases, and activation function to the model
      self.layers.append((weights, bias))
      self.activation_functions.append(activation)


      current_input_size = neurons

      # Addition of hidden layers
      #Initialize the weights and biases for the input layers
      for i, layer in enumerate(input_layers_config):
        neurons = layer['neurons']
        activation = layer['activation']

        # Initialize weights and biases for the current layer
        weights = np.random.randn(current_input_size, neurons)   # Weight initialization
        bias = np.zeros((1, neurons))  # Bias initialization

        # Store layer information
        self.layer_data.append({
          "Layer": f"Hidden Layer {i+1}",
          "Input Neurons": current_input_size,
          "Output Neurons": neurons,
          "Weights": weights.size,
          "Biases": bias.size,
          "Total Parameters": weights.size + bias.size
        })

        # Append weights, biases, and activation function to the model
        self.layers.append((weights, bias))
        self.activation_functions.append(activation)

        # Update current input size for next layer
        current_input_size = neurons

      # Addition of hidden layers
      # Initialize weights and biases for the output layer
      output_neurons = output_layers_config[0]['neurons']
      output_activation = output_layers_config[0]['activation']


      output_weights = np.random.randn(current_input_size, output_neurons)
      output_bias = np.zeros((1, output_neurons))

      self.layer_data.append({
          "Layer": "Output Layer",
          "Input Neurons": current_input_size,
          "Output Neurons": output_neurons,
          "Weights": output_weights.size,
          "Biases": output_bias.size,
          "Total Parameters": output_weights.size + output_bias.size
        })

      # Append output weights, biases, and activation function
      self.layers.append((output_weights, output_bias))
      self.activation_functions.append(output_activation)
    def display_parameters(self):
      """
      Return the parameters of each layer in a DataFrame format.

      Returns:
      - DataFrame: A pandas DataFrame containing parameter details for each layer.
      """
      df = pd.DataFrame(self.layer_data)
      return df
    def feedforward(self, X):
      """
      Perform a forward pass through the neural network.
      Args:
      - X (numpy.ndarray): The input to the network from CNN output.
      Returns:
      - yHat (numpy.ndarray): The output of the fully connected network.
      """
      for i, layer in enumerate(self.layers):
        # Extract weights and biases for the current layer
          weights, bias = layer
        # Matrix multiplication and bias addition
          X = np.dot(X, weights) + bias
          # Apply activation function
          if self.activation_functions[i] == 'relu':
            X = np.maximum(0, X)  # Using np.maximum for ReLU activation
          elif self.activation_functions[i] == 'softmax':
            exps = np.exp(X - np.max(X, axis=1, keepdims=True))
            X = exps / np.sum(exps, axis=1, keepdims=True)  # Softmax activation
          yHat = X
          print('YHAT',yHat)
          
      return yHat
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        n_layers = len(self.layers)
    
        # Store activations and derivatives
        activations = []
        derivatives = []
    
        # Forward pass with stored values
        current_input = X
        activations.append(current_input)
    
        # Forward pass storing intermediate values
        for i, (weights, bias) in enumerate(self.layers):
            Z = np.dot(current_input, weights) + bias
        
            if self.activation_functions[i] == 'relu':
                A = self.relu(Z)
                derivatives.append(self.relu_derivative(Z))
            elif self.activation_functions[i] == 'softmax':
                A = self.softmax(Z)
                derivatives.append(None)  # Special handling for softmax
            
            activations.append(A)
            current_input = A
    
        # Backward pass
        dA = activations[-1] - y  # Derivative of softmax with cross-entropy
    
        for i in range(n_layers - 1, -1, -1):
            if i == n_layers - 1:  # Output layer
                dZ = dA
            else:  # Hidden layers
                dZ = np.dot(dA, self.layers[i + 1][0].T) * derivatives[i]
        
            # Calculate gradients
            dW = np.dot(activations[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
        
            # Update weights and biases
            self.layers[i] = (
                self.layers[i][0] - learning_rate * dW,
                self.layers[i][1] - learning_rate * db
            )
        
            dA = dZ

    def compute_loss(self, yHat, y):
        # Cross-entropy loss
        m = y.shape[0]
        loss = -np.sum(y * np.log(yHat + 1e-9)) / m
        return loss
    def train(self, X_train, y_train, X_val, y_val, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward and backward pass, and weight updates for each batch
            yHat = self.feedforward(X_train)
            loss = self.compute_loss(yHat, y_train)
            print('loss'+loss)
            self.backward(X_train, y_train, learning_rate)
            
            # Validation accuracy check (optional)
            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(val_pred, y_val)
            val_accuracy = self.calculate_accuracy(val_pred, y_val)
            
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    def calculate_accuracy(self, yHat, y):
        # Calculate accuracy based on predictions and true labels
        pred_classes = np.argmax(yHat, axis=1)
        true_classes = np.argmax(y, axis=1)
        return np.mean(pred_classes == true_classes)

    @staticmethod
    def hyperparameter_search(X_train, y_train, X_val, y_val, param_grid,input_layers_config, output_layers_config):
        best_accuracy = 0
        best_params = None
        
        for params in param_grid:
            print(f"Testing configuration: {params}")
            model = NeuralNetwork()
            model.addnetwork(input_size=params['input_size'], input_layers_config=input_layers_config, output_layers_config=output_layers_config)

            
            model.train(
                X_train, y_train, X_val, y_val,
                epochs=params['epochs'],
                learning_rate=params['learning_rate']
            )
            
            # Evaluate on validation set
            val_pred = model.feedforward(X_val)
            val_accuracy = model.calculate_accuracy(val_pred, y_val)
            
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            
            # Track the best configuration
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_params = params
        
        print(f"Best Configuration: {best_params}")
        print(f"Best Validation Accuracy: {best_accuracy:.4f}")
        
        return best_params, best_accuracy        

#create dummy CNN output


dummy_cnn_output = torch.randn(32, 200)  # (batch_size, num_features)
print("Dummy CNN output shape:", dummy_cnn_output.shape)

#Initialize the neural network with flexible layer configuration
#network = NeuralNetwork(input_size=200, input_layers_config=input_layers_config, output_layers_config=output_layers_config)

#output = network.feedforward(dummy_cnn_output)
#print(output)

# Sample usage of hyperparameter_search
param_grid = [
    {'input_size': 784, 'hidden_sizes': [128, 64], 'output_size': 10, 'dropout_rate': 0.2, 
     'activation': 'relu', 'epochs': 10, 'learning_rate': 0.01},
    {'input_size': 784, 'hidden_sizes': [256, 128, 64], 'output_size': 10, 'dropout_rate': 0.3, 
     'activation': 'relu', 'epochs': 10, 'learning_rate': 0.001},
    # Add more configurations as needed
]

# Convert validation set to numpy arrays for easy handling with the NeuralNetwork class
# Using DataLoader for batch processing if needed
val_loader = DataLoader(new_validation_dataset, batch_size=len(new_validation_dataset))
X_val, y_val = next(iter(val_loader))  # Get entire validation set in one batch
X_val = X_val.view(len(X_val), -1).numpy()  # Flatten and convert to numpy array
y_val = torch.nn.functional.one_hot(y_val, num_classes=10).numpy()  # One-hot encode labels

# Define parameter grid for hyperparameter search
param_grid = [
    {'input_size': X_val.shape[1], 'epochs': 10, 'learning_rate': 0.01},
    {'input_size': X_val.shape[1], 'epochs': 20, 'learning_rate': 0.001},
    {'input_size': X_val.shape[1], 'epochs': 30, 'learning_rate': 0.005}
]

# Perform hyperparameter search
best_params, best_accuracy = NeuralNetwork.hyperparameter_search(new_train_dataset, new_test_dataset, X_val, y_val, param_grid, input_layers_config, output_layers_config)

print(f"Best hyperparameters: {best_params}")
print(f"Best validation accuracy: {best_accuracy:.4f}")


#Initialize the neural network with flexible layer configuration
nn_network = NeuralNetwork()
cnn_output_np = cnn_output.detach().numpy()
cnn_output_np=nn_network.flatten(cnn_output_np)  # Flatten the CNN output
nn_network.addnetwork(input_size=cnn_output_np.shape[1], input_layers_config=input_layers_config, output_layers_config=output_layers_config)

output = nn_network.feedforward(cnn_output_np)

# Retrieve and display the parameter DataFrame
df = nn_network.display_parameters()
print("Parameter Table for Neural Network:")
print(df)
#add the code for hyperparameter_search


