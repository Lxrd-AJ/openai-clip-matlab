function net = setLearnRate(net, factor)
    learnables = net.Learnables;
    numLearnables = size(learnables,1);
    for i = 1:numLearnables
        layerName = learnables.Layer(i);
        parameterName = learnables.Parameter(i);
        
        net = setLearnRateFactor(net,layerName,parameterName,factor);
    end
end