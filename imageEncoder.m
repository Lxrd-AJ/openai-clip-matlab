function net = imageEncoder()
    resnet = imagePretrainedNetwork("resnet50", Weights="pretrained");
    net = networkLayer(resnet, OutputNames="avg_pool");
end