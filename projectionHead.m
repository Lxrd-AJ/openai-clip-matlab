function net = projectionHead(opts)
    arguments
        opts.Name = "proj"
        opts.ProjectionDims = 256
    end

    net = dlnetwork();

    layers = [
        fullyConnectedLayer(opts.ProjectionDims, Name="fc1")
        geluLayer()
        fullyConnectedLayer(opts.ProjectionDims, Name="fc2")
    ];
    net = addLayers(net, layers);

    % tail = [
    %     additionLayer(2, Name="add")
    %     layerNormalizationLayer()
    % ];
    % net = addLayers(net, tail);
    % 
    % net = connectLayers(net, "fc1", "add/in1");
    % net = connectLayers(net, "fc2", "add/in2");
    net = networkLayer(net, Name=opts.Name);
end