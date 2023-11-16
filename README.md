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