function [imagesBatch, inputIDs, attentionMask, segmentIDs] = processMiniBatch(images, tokenisedCaption, ~) 
    keyboard;
    segmentIDs = ones(size(tokenisedCaption));

    % [~, tokenizer] = bert("Model","tiny");
    % paddingValue = tokenizer.PaddingCode;
    paddingValue = 1; % Hard coded bert tokeniser padding code

    [inputIDs, attentionMask] = padsequences(tokenisedCaption, 2, "PaddingValue", paddingValue);
    segmentIDs = padsequences(segmentIDs, 2, "PaddingValue", paddingValue);
    
    % TODO: Image transformation and concatenation
    keyboard;
end