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


# Dataset.py
Bilingual Dataset for Sequence-to-Sequence Models
The BilingualDataset class is a custom PyTorch Dataset designed to handle bilingual data, making it particularly useful for sequence-to-sequence (seq2seq) models such as machine translation systems. This dataset class facilitates the processing of source and target language pairs, applying tokenization, and preparing data for training seq2seq models.

Features<br>
`Bilingual Sentence Pairs`: Handles datasets containing pairs of sentences in two different languages, referred to as the source and target languages.<br>
`Tokenization`: Integrates tokenization for both source and target languages, converting text sentences into sequences of tokens.<br>
`Fixed Sequence Length`: Supports padding and truncation to ensure all sequences have a uniform length.<br>
`Special Tokens`: Automatically adds special tokens such as Start-of-Sentence (SOS), End-of-Sentence (EOS), and padding (PAD) tokens to sequences.<br>
`Attention Masks`: Generates attention masks for both encoder and decoder inputs to ignore padding tokens during the attention mechanism<br>




## Contributing
Contributions to improve the library are welcome. Please ensure to follow the coding standards and submit a pull request for review.

## License
This library is open-sourced under the MIT license.


