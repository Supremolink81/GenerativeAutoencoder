import torch
import numpy as np
from torch import utils
from torch.utils import data
import random

class DenoisifiedAutoEncoder(torch.nn.Module):
    """
    Generative Denoisified Autoencoder (DAE) implementation using PyTorch.

    For construction, accepts a list of layer sizes, a list of encoder
    activation functions, and a list of decoder activation functions.

    Implemented according to the following paper: https://arxiv.org/pdf/1305.6663.pdf
    """
    
    corruption_distribution : callable

    def __init__(
        self, 
        corruption_distribution : callable,
        encoder_and_decoder_layer_sizes : 'list[int]', 
        encoder_activation_functions : list,
        decoder_activation_functions : list,
    ):

        # torch.nn.Module superclass constructor
        super().__init__()

        self.check_assertions( 
            encoder_and_decoder_layer_sizes, 
            encoder_activation_functions, 
            decoder_activation_functions, 
        )

        self.construct_autoencoder(
            corruption_distribution,
            encoder_and_decoder_layer_sizes,
            encoder_activation_functions,
            decoder_activation_functions
        )

    # ---------- HELPER FUNCTIONS FOR FORWARD AND BACK PROPAGATION ----------

    def forward(
            self, 
            data, 
        ):
        # Corrupt data 
        noise_array = self.corruption_distribution(tuple(data.shape))
        noisified_data = data + torch.from_numpy(noise_array)

        # Pass data through autoencoder
        encoded = self.encoder(noisified_data)
        decoded = self.decoder(encoded)

        return decoded
    
    def backpropagate(
        self,
        optimizer : torch.optim.Optimizer,
        loss,
    ):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ---------------------- CONSTRUCTOR HELPER FUNCTIONS ----------------------

    def check_assertions(
            self,
            encoder_and_decoder_layer_sizes : 'list[int]', 
            encoder_activation_functions : list,
            decoder_activation_functions : list,
    ):
        """
        Check for the following:

        - the number of activation functions for the encoder
        and decoder are equal

        - the number of activations functions for the encoder and decoder
        are one less than the amount of layers for the encoder and decoder
        (each adjacent pair of layers needs an activation function)

        - there are at least 2 layers entered for the
        encoder and decoder
        """
        assert len(decoder_activation_functions)-1 == len(encoder_activation_functions)-1 == len(encoder_and_decoder_layer_sizes), "Activation function lists and layer size list must be the same size."
        assert len(encoder_and_decoder_layer_sizes) > 1, "Autoencoder layer sizes list must contain at least 1 element"
            
    def construct_autoencoder(
            self,
            corruption_distribution : callable,
            encoder_and_decoder_layer_sizes : list, 
            encoder_activation_functions : list,
            decoder_activation_functions : list,
    ) -> None:
        """
        Constructs an autoencoder using a set of layer sizes and 
        activation functions for each pair of adjacent layers.

        The autoencoder is constructed using 
        """
        # Initializing member variables
        self.corruption_distribution = corruption_distribution
        self.encoder = torch.nn.Sequential()
        self.decoder = torch.nn.Sequential()
        number_of_layers = len(encoder_and_decoder_layer_sizes)

        for i in range(number_of_layers-1):

            # Encoder helper variables and construction (add each adjacent pair of
            # layers to the encoder, as well as the activation function that connects them)
            current_encoder_layer_size = encoder_and_decoder_layer_sizes[i]
            next_encoder_layer_size = encoder_and_decoder_layer_sizes[i+1]
            encoder_layer_activation_function = encoder_activation_functions[i]
            self.encoder.append(torch.nn.Linear(current_encoder_layer_size, next_encoder_layer_size))
            self.encoder.append(encoder_layer_activation_function)

            # Decoder helper variables (have to get layer sizes backwards because
            # decoder needs to convert back to the original input's size) and construction (add each adjacent pair of
            # layers to the encoder, as well as the activation function that connects them)
            current_decoder_layer_size = encoder_and_decoder_layer_sizes[number_of_layers-i-1]
            next_decoder_layer_size = encoder_and_decoder_layer_sizes[number_of_layers-i-2]
            decoder_layer_activation_function = decoder_activation_functions[i]
            self.decoder.append(torch.nn.Linear(current_decoder_layer_size, next_decoder_layer_size))
            self.decoder.append(decoder_layer_activation_function)

    # -------- HELPER FUNCTIONS FOR TRAINING AND TESTING THE AUTOENCODER --------
    
    def train_autoencoder(
            self,
            epochs : int,
            loss_function,
            optimizer : torch.optim.Optimizer,
            training_set,
            p,
    ) -> None:
        """
        Trains the autoencoder on a set of training data.
        
        Allows the user to pass in a custom loss function and
        optimizer for maximum flexibility over the model.
        """
        training_loader = data.DataLoader(dataset=training_set, batch_size=32, training=True, shuffle=True)
        for e in range(epochs):
            print(f"Epoch {e + 1}: ")
            for (feature_vector, _) in training_loader:

                # Get output of training example and loss value
                reconstructed_feature_vector = self.forward(feature_vector)
                loss = loss_function(reconstructed_feature_vector, feature_vector)
                self.backpropagate(optimizer, loss)
                
                # Initialize dataset for Markov chain of training example
                chain = self.generate_corrupted_training_examples(feature_vector, p)
                for example in chain:
                    reconstructed_feature_vector = self.forward(example)
                    loss = loss_function(reconstructed_feature_vector, feature_vector)
                    self.backpropagate(optimizer, loss)

    def validate_autoencoder(
        self,
        loss_function,
        validation_set,
    ) -> None:
        """
        Tests a set of validation data on the autoencoder, which
        is assumed to be trained on a set of training data.
        """
        validation_loader = data.DataLoader(dataset=validation_set, batch_size=32, training=False, shuffle=False)
        for testing_example in validation_loader:
            testing_output = self.forward(testing_example)
            loss = loss_function(testing_output, testing_example)

    def generate_corrupted_training_examples(
            self, 
            training_example, 
            p = 0.5
    ) -> list:
        """
        Walkback algorithm to train a Markov chain of training examples
        that are corrupted versions of the original example.

        This allows the autoencoder to catch clusters of examples
        that are difficult to denoise, providing a more
        robust and efficient solution to overfitting.
        """
        new_example, example_list = training_example.clone(), []
        while True:
            # Get Gaussian noise for training example (need to convert PyTorch tensor shape to 
            # tuple because default type is Tensor.Size object)
            noise_array = self.corruption_distribution(tuple(new_example.shape))
            new_example += torch.from_numpy(noise_array)
            u = random.uniform(0,1)
            new_example.append(example_list)
            if u > p:
                return example_list
            new_example = self.forward(new_example)