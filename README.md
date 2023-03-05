<h1>GenerativeAutoencoder</h1>
<h2>Description</h2>

Denoising Autoencoder (DAE) in PyTorch generalized as a generative model.  

Allows the user to set a custom optimizer, corruption function,  
activation functions, and layer sizes.

Generates new training examples through a "walkback" algorithm, allowing the  
model to prevent overfitting without the need for more training samples  
or more robust noise.

For more detailed information, consult the following paper: https://arxiv.org/abs/1305.6663
