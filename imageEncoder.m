function net = imageEncoder(opts)
    arguments
        opts.LearnRate = 0
    end
    resnet = imagePretrainedNetwork("resnet50", Weights="pretrained");
    net = setLearnRate(resnet, opts.LearnRate);
    net = networkLayer(net, Name="image_encoder", OutputNames="avg_pool");
end