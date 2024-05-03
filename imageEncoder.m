function net = imageEncoder()
    resnet = imagePretrainedNetwork("resnet50", Weights="pretrained");
    net = networkLayer(resnet, Name="image_encoder", OutputNames="avg_pool");
end