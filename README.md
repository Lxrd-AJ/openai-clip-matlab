# openai-clip-matlab
[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=Lxrd-AJ/openai-clip-matlab&file=playground.mlx)

MATLAB implementation of the OpenAI CLIP deep learning model

## Flickr Dataset
See https://hockenmaier.cs.illinois.edu/8k-pictures.html 
Data sources for download
* Flickr 8K https://github.com/goodwillyoga/Flickr8k_dataset or see https://github.com/jbrownlee/Datasets/blob/master/Flickr8k_Dataset.names 
* Flickr 8K https://www.kaggle.com/datasets/adityajn105/flickr8k/data 

# Other notes
* Use https://uk.mathworks.com/help/matlab/ref/memmapfile.html to store & query the image embeddings for fast search

# TODO (Training)
- [x] Design a smaller model (use Bert tiny and design a smaller image encoder from an existing pretrained image model - use squeezenet)
    - [x] Allow the encoder models to learn but with a smaller learning rate
- [x] Use [SEP] token from bert rather than [CLS] token
- [x] Allow the model to learn the logits scaling 
- [ ] Support training on the train, validation and test sets
- [ ] Follow model design and training guides in Section 2.4 & 2.5
    - [ ] Use cosine schedule
    - [ ] Clip logits scaling temperature parameter to 100 max
- [ ] Move image resizing outside of the `processMiniBatch` function and into a transform function for the datastore
- [ ] Upgraded datastore class: Use the provided train, validation and test sets.
- [ ] Calculate accuracy metric: `argmax(logits) == targets`
- [ ] Save the model at different checkpoints during training
- [ ] Train on Flickr30k dataset


# TODO (Model Interface)
- [ ] Wrapper class around the CLIP model
    - [ ] See API in https://github.com/openai/CLIP?tab=readme-ov-file#api 
        - [x] Encode images
        - [x] Get softmax and logit scores for a batch of (image, text) pair
    - [ ] Find the top-k images that match a given query
- [ ] Comparison against CIFAR-10(100)
    - https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb 
    - https://uk.mathworks.com/help/deeplearning/ug/train-residual-network-for-image-classification.html 
- [ ] Front end GUI for interfacing with the model (using uicomponentcontainer)
- [ ] Index an existing folder
- [ ] Run indexing in `backgroundPool` 