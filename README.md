# Image Captioning
 
## Acknowledgement

This work builds on the image captioning model from [keras](https://keras.io/examples/vision/image_captioning/).

# Image Captioning with Transformer Model

This repository contains the implementation of an image captioning model using the Transformer architecture. The project aims to develop a model that can generate descriptive captions from images.

## Overview

The image captioning model is built using the Transformer architecture. This powerful model leverages the power of deep learning to understand the content of images and generate relevant captions.

## Model Architecture

The architecture is based on the Transformer model, and it includes:

- **Embedding Layer:** This layer adds positional information to the input tensor. This is important for Transformers, as they otherwise have no way to obtain such positional information.
- **Encoder Blocks:** These implement single encoder blocks from the Transformer model architecture.
- **Decoder Blocks:** These implement single decoder blocks from the Transformer model architecture.
- **Model Combination:** All the components of the Transformer model architecture are combined to implement the final model.

## Usage

Clone this repository to your local system and install the necessary dependencies. You should have Python and the required libraries installed on your system to run the model. In addition, download the saved model along with the text files saved in pickle. 

## Contributions

Contributions to this repository are welcome. If you find a bug or think of a feature that would benefit the project, please open an issue or submit a pull request.

## Acknowledgement

This work builds on the image captioning model example provided by [Keras](https://keras.io/examples/vision/image_captioning/). We thank the Keras team for providing the base model and inspiring this project.

## Contact

For any queries or suggestions, please open an issue on this repository.
