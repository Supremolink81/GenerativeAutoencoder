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
        corruption_distribution: callable,
        encoder_layer_sizes: 'list[int]', 
        decoder_layer_sizes: 'list[int]', 
        encoder_activation_functions: list,
        decoder_activation_functions: list,
    ):

        # calls superclass constructor to initialize 
        # helper member variables, such as encoder
        # and decoder
        super().__init__()

        self.check_assertions( 
            encoder_layer_sizes, 
            decoder_layer_sizes, 
            encoder_activation_functions, 
            decoder_activation_functions, 
        )

        self.construct_autoencoder(
            corruption_distribution,
            encoder_layer_sizes, 
            decoder_layer_sizes, 
            encoder_activation_functions,
            decoder_activation_functions,
        )

    def forward(
        self, 
        data: data.Dataset, 
    ) -> torch.Tensor:
        noise_array: np.array = self.corruption_distribution(tuple(data.shape))
        noisified_data: torch.Tensor = data + torch.from_numpy(noise_array)

        encoded: torch.Tensor = self.encoder(noisified_data)
        decoded: torch.Tensor = self.decoder(encoded)

        return decoded
    
    def backpropagate(
        self,
        optimizer : torch.optim.Optimizer,
        loss,
    ) -> None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def check_assertions(
        self,
        encoder_layer_sizes: 'list[int]', 
        decoder_layer_sizes: 'list[int]', 
        encoder_activation_functions: list,
        decoder_activation_functions: list,
    ) -> None:
        """
        Checks for the following:

        - the number of activation functions for the encoder 
        is one less than the number of layers (and same for
        the decoder)

        - there are at least 2 layers each for 
        the encoder and the decoder

        - the last layer of the encoder and the first
        layer of the decoder are the same size

        - the first layer of the encoder and the last
        layer of the decoder are the same size
        """
        assert len(decoder_activation_functions)-1 == len(decoder_layer_sizes), "Decoder activation function list size must be 1 less than decoder layer sizes list size."
        assert len(encoder_activation_functions)-1 == len(encoder_layer_sizes), "Encoder activation function list size must be 1 less than encoder layer sizes list size."
        assert len(decoder_layer_sizes) > 1, "Decoder layer sizes list must contain at least 2 elements."
        assert len(encoder_layer_sizes) > 1, "Encoder layer sizes list must contain at least 2 elements."
        assert encoder_layer_sizes[0] == decoder_layer_sizes[-1], "The first layer of the encoder and the last layer of the decoder must be the same size."
        assert encoder_layer_sizes[-1] == decoder_layer_sizes[0], "The last layer of the encoder and the first layer of the decoder must be the same size."
            
    def construct_autoencoder(
            self,
            corruption_distribution: callable,
            encoder_layer_sizes: 'list[int]', 
            decoder_layer_sizes: 'list[int]', 
            encoder_activation_functions: list,
            decoder_activation_functions: list,
    ) -> None:
        """
        Constructs an autoencoder using a corruption function, a set of layer
        sizes, and an activation functions for each pair of adjacent layers.
        """
        self.corruption_distribution: callable = corruption_distribution
        self.encoder = torch.nn.Sequential()
        self.decoder = torch.nn.Sequential()
        number_of_encoder_layers: int = len(encoder_layer_sizes)
        number_of_decoder_layers: int = len(decoder_layer_sizes)

        for i in range(number_of_encoder_layers-1):
            
            current_encoder_layer_size: int = encoder_layer_sizes[i]
            next_encoder_layer_size: int = encoder_layer_sizes[i+1]
            encoder_layer_activation_function = encoder_activation_functions[i]
            self.encoder.append(torch.nn.Linear(current_encoder_layer_size, next_encoder_layer_size))
            self.encoder.append(encoder_layer_activation_function)

        for i in range(number_of_decoder_layers-1):

            # have to get layer sizes backwards because decoder needs to convert back to the original input's size
            # (i.e. do what the encoder did in reverse)
            current_decoder_layer_size: int = decoder_layer_sizes[i]
            next_decoder_layer_size: int = decoder_layer_sizes[i+1]
            decoder_layer_activation_function = decoder_activation_functions[i]
            self.decoder.append(torch.nn.Linear(current_decoder_layer_size, next_decoder_layer_size))
            self.decoder.append(decoder_layer_activation_function)
    
    # loss functions don't have type hints because no
    # base class for loss functions exists in PyTorch

    def train_autoencoder(
            self,
            epochs: int,
            loss_function,
            optimizer: torch.optim.Optimizer,
            training_set: data.Dataset,
            p: float = 0.5,
    ) -> None:
        """
        Trains the autoencoder on a set of training data.
        
        Allows the user to pass in a custom loss function and
        optimizer for maximum flexibility over the model.
        """
        training_loader = data.DataLoader(dataset=training_set, batch_size=32, training=True, shuffle=True)
        for e in range(epochs):
            print(f"Epoch {e + 1}: \n")
            for (feature_vector, _) in training_loader:

                reconstructed_feature_vector: torch.Tensor = self.forward(feature_vector)
                loss = loss_function(reconstructed_feature_vector, feature_vector)
                self.backpropagate(optimizer, loss)
                print(f"Loss: {loss}")
                
                chain: list = self.generate_corrupted_training_examples(feature_vector, p)
                for example in chain:
                    reconstructed_feature_vector = self.forward(example)
                    loss = loss_function(reconstructed_feature_vector, feature_vector)
                    print(f"Loss: {loss}")
                    self.backpropagate(optimizer, loss)
                
                # newline for formatting
                print()

    def validate_autoencoder(
        self,
        loss_function,
        validation_set: data.Dataset,
    ) -> None:
        """
        Tests a set of validation data on the autoencoder, which
        is assumed to be trained on a set of training data.

        (Currently does nothing, will do something in
        a future release.)
        """
        validation_loader = data.DataLoader(dataset=validation_set, batch_size=32, training=False, shuffle=False)
        for testing_example in validation_loader:
            testing_output = self.forward(testing_example)
            loss = loss_function(testing_output, testing_example)

    def generate_corrupted_training_examples(
            self, 
            training_example: torch.Tensor, 
            p: float = 0.5,
    ) -> list:
        """
        Walkback algorithm to train a Markov chain of training examples
        that are corrupted versions of the original example.

        This allows the autoencoder to catch clusters of examples
        that are difficult to denoise, providing a more
        robust and efficient solution to overfitting.
        """
        # if p = 1 we have an infinite loop
        assert p >= 1 or p < 0, ("A probability value must be in between 0 and 1 (not including 1).")
        new_example: torch.Tensor = training_example.clone()
        example_list: list = []
        u: float = 0

        while u <= p:
            noise_array: np.array = self.corruption_distribution(tuple(new_example.shape))
            new_example += torch.from_numpy(noise_array)
            u = random.uniform(0,1)
            example_list.append(new_example)
            new_example = self.forward(new_example)
        return example_list