# openai-clip-matlab
MATLAB implementation of the OpenAI CLIP deep learning model

## Flickr Dataset
See https://hockenmaier.cs.illinois.edu/8k-pictures.html 
Data sources for download
* Flickr 8K https://github.com/goodwillyoga/Flickr8k_dataset or see https://github.com/jbrownlee/Datasets/blob/master/Flickr8k_Dataset.names 
* Flickr 8K https://www.kaggle.com/datasets/adityajn105/flickr8k/data 

# Other notes
* Use https://uk.mathworks.com/help/matlab/ref/memmapfile.html to store & query the image embeddings for fast search

# TODO (Training)
- [ ] Design a smaller model (use Bert tiny and design a smaller image encoder from an existing pretrained image model - use squeezenet)
    - [ ] Allow the encoder models to learn but with a smaller learning rate
- [ ] Save the model at different checkpoints during training
- [ ] Follow model design and training guides in Section 2.4 & 2.5
    - [ ] Use cosine schedule
    - [ ] Clip logits scaling temperature parameter to 100 max
- [ ] Move image resizing outside of the `processMiniBatch` function and into a transform function for the datastore
- [ ] Upgraded datastore class: Use the provided train, validation and test sets.


# TODO (Model Interface)
- [ ] Wrapper class around the CLIP model
- [ ] Front end GUI for interfacing with the model (using uicomponentcontainer)
- [ ] Index an existing folder
- [ ] Run indexing in `backgroundPool` 