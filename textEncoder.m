function net = textEncoder(opts)
    arguments
        opts.LearnRate = 0
        opts.BertModel = "tiny" % 4.3M learnables
    end
    [net, ~] = bert(Model=opts.BertModel);
    clsEmbeddingLayer = functionLayer(@(x) x(:,:,end), Name='clsTokenEmbedding'); % Takes out the 1st element (CLS Token) along the time dimension
    net = dlnetwork([
        networkLayer(net, Name="bert_model")
        clsEmbeddingLayer
    ], Initialize=false);
    net = setLearnRate(net, opts.LearnRate);
    net = networkLayer(net, Name="bert_encoder");
end