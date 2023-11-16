# Intro to Neural Networks - Fall 2023
JHU Engineering for Professionals

## Setup
To setup this project, you'll need to install Anaconda, a popular Python package manager.

### Setting up Anaconda
Follow [these instructions](https://docs.anaconda.com/free/anaconda/install/index.html) to install Anaconda for your operating system.

A great thing to do after this is to install a library called Mamba. Anaconda has a horrible time trying to figure out which packages are good for your environment (what's called solving your environment) and Mamba uses a different algorithm to attempt to figure out those packages faster (which it often does).

This is completely optional and is more to help you save time than anything else.

Here's how to install it:
`$ conda install -n base conda-libmamba-solver`

To set it as your default solver:
`$ conda config --set solver libmamba`

#### Setting up the Anaconda environment for the project

Run `$ conda env create -f environment.yml`

Hit y for the prompts and you should be good to go!


## Running the Code

The main classes to create a neural network are the Layer and Network classes. 

The code below creates a network with two layers, one with 2 inputs, 2 outputs, then the last one with 2 inputs and 1 output. Read the comments below to learn more about the classes.
```py
def get_network(use_biases: bool) -> Network:
    return (Network()   # Create a network object to hold our layers.
    .add_layer(     # add a layer to the network 
        Layer(num_in_features=2,    # This layer takes two inputs.
              num_out_features=2,   # This layer has two outputs.
              is_hidden_layer=True, # This layer is a hidden layer as it is not the final output of a network. 
              initial_weights=[np.ones(2) * 0.3, np.ones(2) * 0.3],  # These are the initial weights to the network, one for each neuron in the order that the inputs are to be processed.
              use_biases=use_biases,    # This is a boolean value to tell the layer whether to use and train biases during its use.  
              initial_biases=[0, 0]))   # The initial values of the bias for both neurons.
    .add_layer(
        Layer(num_in_features=2,    # Same as before
              num_out_features=1,
              is_hidden_layer=False,    # Since this layer is the final output of the network, this is set to false.
              initial_weights=[np.ones(2) * 0.8],   # The initial weights. There's only one since there's only a single output for the layer.
              use_biases=use_biases,
              initial_biases=[0, 0])))
```

To do the FFBP algorithm, do the following:
```py 
for epoch in range(num_epochs):     # Go through each training epoch we're required to do.
    for i in range(X.shape[0]):   # For each piece of data we need to process. 
        curr_out: np.ndarray = network.feedforward(X[i, :])   # Get the current output from the network. 
        network.backprop(lr, y[i, :]) # Train the network using the learning rate and the desired output.
```

