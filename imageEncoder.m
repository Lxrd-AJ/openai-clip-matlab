function net = imageEncoder(opts)
    arguments
        opts.LearnRate = 0
    end
    % resnet = imagePretrainedNetwork("resnet50", Weights="pretrained");
    % net = setLearnRate(resnet, opts.LearnRate);
    % net = networkLayer(net, Name="image_encoder", OutputNames="avg_pool");

    % `net` has about 750k learnables
    net = imagePretrainedNetwork("squeezenet", Weights="pretrained");
    layersToRemove = [
        "drop9"
        "conv10"
        "relu_conv10"
        "pool10"
        "prob"
        "prob_flatten"
    ];
    net = removeLayers(net, layersToRemove);
    net = addLayers(net, flattenLayer(Name="flatten"));
    net = connectLayers(net, "fire9-concat", "flatten");
    net = initialize(net);
    net = setLearnRate(net, opts.LearnRate);
    net = networkLayer(net, Name="image_encoder", OutputNames="flatten");
end