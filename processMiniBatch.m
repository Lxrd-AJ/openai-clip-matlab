function [imagesBatch, inputIDs, attentionMask, segmentIDs] = processMiniBatch(images, tokenisedCaption, ~) 
    % NB: Hard code the `PaddingValue` for now
    % [~, tokenizer] = bert("Model","tiny");
    % paddingValue = tokenizer.PaddingCode;
    paddingValue = 1; % Hard coded bert tokeniser padding code
    
    [inputIDs, attentionMask] = padsequences(tokenisedCaption, 2, "PaddingValue", paddingValue); % Returns in CTB format
    inputIDs = permute(inputIDs, [1 3 2]);
    attentionMask = permute(attentionMask, [1 3 2]);
    segmentIDs = ones(size(inputIDs)); % The `segmentIDs` are always 1, constraint imposed by the `bert` language model
    
    % TODO: Move image resizing outside into a transform datastore
    imagesBatch = cellfun(@(x) imresize(x, [224 224]), images, UniformOutput=false);
    imagesBatch = cat(4, imagesBatch{:});

    if canUseGPU
        imagesBatch = gpuArray(imagesBatch);
        inputIDs = gpuArray(inputIDs);
        attentionMask = gpuArray(attentionMask);
        segmentIDs = gpuArray(segmentIDs);
    end
end