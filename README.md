# Transformer Components in PyTorch
This README outlines the implementation of various components used in the Transformer architecture, as described in the paper "Attention is All You Need" by Vaswani et al. The implementation is done using PyTorch, a popular deep learning library. The components covered include Input Embedding, Positional Embedding, Layer Normalization, FeedForward Block, Multi-Head Attention Block, and Residual Connection. These components are foundational for building Transformer models for tasks like language translation, text summarization, and more.

## Dependencies
1. PyTorch
2. Math <br>
Ensure you have PyTorch installed in your environment to use these components. You can install PyTorch by following the instructions on the official website.

![alt text](assets/images/transfomer_architecture.webp)
# Features
`Input Embedding`: Converts token indices into dense vectors of a specified size. <br>
`Positional Embedding`: Adds positional information to the input embeddings, allowing the model to recognize word order.<br>
`Layer Normalization`: Applies normalization across the features for each data point in a batch to stabilize the learning process.<br>
`FeedForward Block`: Implements the feedforward neural network layer within the Transformer model.<br>
`Multi-Head Attention`: Allows the model to jointly attend to information from different representation subspaces at different positions.<br>
`Residual Connection`: Facilitates the training of deep networks by allowing gradients to flow through the architecture more effectively.<br>

## Contributing
Contributions to improve the library are welcome. Please ensure to follow the coding standards and submit a pull request for review.

## License
This library is open-sourced under the MIT license.