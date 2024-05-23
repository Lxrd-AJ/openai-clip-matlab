classdef CLIP < handle
    % CLIP

    properties(Access=private)
        Net
        ImageInputSize
        Temperature
        BertTokenizer
    end

    methods
        function this = CLIP(dlnet, opts)
            arguments
                dlnet
                opts.ImageInputSize = [224 224]
                opts.Temperature = 100
            end
            this.Net = dlnet;
            this.ImageInputSize = opts.ImageInputSize;
            this.Temperature = opts.Temperature;
            [~, this.BertTokenizer] = bert();
        end

        % encodeImageAt
        % Encode images at the supplied URLs
        function imageEmbeddings = encodeImagesAt(this, imagePaths)
            numImages = numel(imagePaths);
            % NB: I can't seem to exclude the bert part of the network from the predict graph
            dummyTokens = dlarray(randi(10, [1 numImages 10]), 'CBT');
            dummyAttentionMasks = dlarray(ones(size(dummyTokens)), 'CBT');
            dummySegmentIDs = dlarray(ones(size(dummyTokens)), 'CBT');

            % Read in the images
            images = arrayfun(@(x) imread(x), imagePaths, UniformOutput=false);
            imagesBatch = cellfun(@(x) imresize(x, this.ImageInputSize), images, UniformOutput=false);
            imagesBatch = cat(4, imagesBatch{:});
            imagesBatch = dlarray(single(imagesBatch), "SSCB");

            [imageEmbeddings, ~] = predict(this.Net, imagesBatch, dummyTokens, dummyAttentionMasks, dummySegmentIDs);
            imageEmbeddings = imageEmbeddings ./ vecnorm(imageEmbeddings, 2, 1); % Along the `C` dimension in `CB`
            imageEmbeddings = stripdims(imageEmbeddings); 
        end

        function textEmbeddings = encodeText(this, textToEncode)
            tokens = encode(this.BertTokenizer, textToEncode);
            numBatch = numel(textToEncode);
            paddingValue = this.BertTokenizer.PaddingCode;
            [tokens, attentionMask] = padsequences(tokens, 2, "PaddingValue", paddingValue); % Returns in CTB format
            tokens = permute(tokens, [1 3 2]); % Change to CBT format
            attentionMask = permute(attentionMask, [1 3 2]);
            segmentIDs = ones(size(tokens)); % The `segmentIDs` are always 1, constraint imposed by the `bert` language model

            dummyImage = dlarray(randn([this.ImageInputSize 3]), "SSC");

            [~, textEmbeddings] = predict(this.Net, dummyImage, tokens, attentionMask, segmentIDs);
            textEmbeddings = textEmbeddings ./ vecnorm(textEmbeddings, 2, 1);
            textEmbeddings = stripdims(textEmbeddings);
        end
    end
end