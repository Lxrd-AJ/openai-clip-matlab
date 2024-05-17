function [loss, gradientsNet, gradientTemp, state] = modelLoss(net, images, inputIDs, attentionMasks, segmentIDs, logTemperature)
    [imageEmbeddings, textEmbeddings, state] = forward(net, images, inputIDs, attentionMasks, segmentIDs);
    % Remove the trailing `T` dimension from `textEmbeddings` 
    textEmbeddings = squeeze(textEmbeddings);
    % Normalise the embeddings
    nImEmbeddings = imageEmbeddings ./ vecnorm(imageEmbeddings, 2, 1);
    nTextEmbeddings = textEmbeddings ./ vecnorm(textEmbeddings, 2, 1);

    % Remove the dimension labels so that transpose ops can be performed
    nImEmbeddings = stripdims(nImEmbeddings);
    nTextEmbeddings = stripdims(nTextEmbeddings);
    
    % Construct the target distribution
    numImages = size(imageEmbeddings, 2); % CB
    targets = onehotencode(1:numImages, 1, 'ClassNames', 1:numImages);

    logits = (nImEmbeddings' * nTextEmbeddings) * exp(logTemperature);
    probsImText = softmax(logits, 'DataFormat', 'SC');
    probsTextIm = softmax(logits', 'DataFormat', 'SC');

    lossImages = crossentropy(probsImText, targets, 'DataFormat', 'SC');
    lossText = crossentropy(probsTextIm, targets, 'DataFormat', 'SC');

    loss = (lossImages + lossText) / 2;

    gradientsNet = dlgradient(loss, net.Learnables);
    gradientTemp = dlgradient(loss, logTemperature);
end