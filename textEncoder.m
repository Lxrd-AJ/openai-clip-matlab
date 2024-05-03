function net = textEncoder()
    [net, ~] = bert();
    net = networkLayer(net);
end