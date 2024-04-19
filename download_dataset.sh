#!/bin/bash

wget -nc https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
wget -nc https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

mkdir -p ./flickr-dataset

unzip Flickr8k_Dataset.zip -d ./flickr-dataset
unzip Flickr8k_text.zip -d ./flickr-dataset