function net = textEncoder()
    [net, ~] = bert();
    clsEmbeddingLayer = functionLayer(@(x) x(:,:,1), Name='clsTokenEmbedding'); % Takes out the 1st element (CLS Token) along the time dimension
    net = dlnetwork([
        networkLayer(net, Name="bert_model")
        clsEmbeddingLayer
    ], Initialize=false);
    net = networkLayer(net, Name="bert_encoder");
end