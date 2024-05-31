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

        function [probs, logits] = predict(this, imagePaths, textToEncode)
            tokens = encode(this.BertTokenizer, textToEncode);

            if isscalar(textToEncode)
                tokens = dlarray(tokens{1}, "CTB"); % This will re-arrange to 'CBT' 
                attentionMask = dlarray(ones(size(tokens)), 'CBT');
            else
                paddingValue = this.BertTokenizer.PaddingCode;
                [tokens, attentionMask] = padsequences(tokens, 2, "PaddingValue", paddingValue); % Returns in CTB format
                tokens = dlarray(permute(tokens, [1 3 2]), "CBT"); % Change to CBT format
                attentionMask = dlarray(permute(attentionMask, [1 3 2]), "CBT");
            end
            segmentIDs = dlarray(ones(size(tokens)), 'CBT'); % The `segmentIDs` are always 1, constraint imposed by the `bert` language model

            images = iPrepareImages(imagePaths, 'ResizeTo', this.ImageInputSize);

            [imageEmbeddings, textEmbeddings] = predict(this.Net, images, tokens, attentionMask, segmentIDs);
            % Remove trailing `T` dimension from `textEmbeddings`
            textEmbeddings = squeeze(textEmbeddings);

            % Normalise the embeddings
            nImEmbeddings = imageEmbeddings ./ vecnorm(imageEmbeddings, 2, 1);
            nTextEmbeddings = textEmbeddings ./ vecnorm(textEmbeddings, 2, 1);

            % Remove the dimension labels so that transpose ops can be performed
            nImEmbeddings = stripdims(nImEmbeddings);
            nTextEmbeddings = stripdims(nTextEmbeddings);
            logits = (nImEmbeddings' * nTextEmbeddings) * this.Temperature;
            
            probs = softmax(logits, "DataFormat", "CS");
        end

        % function textEmbeddings = encodeText(this, textToEncode)

        %     dummyImage = dlarray(randn([this.ImageInputSize 3]), "SSC");

        %     [~, textEmbeddings] = predict(this.Net, dummyImage, tokens, attentionMask, segmentIDs);
        %     textEmbeddings = textEmbeddings ./ vecnorm(textEmbeddings, 2, 1);
        %     textEmbeddings = stripdims(textEmbeddings);
        % end
    end
end

function imagesBatch = iPrepareImages(imagePaths, opts)
    arguments
        imagePaths
        opts.ResizeTo = [224 224]
    end
    images = arrayfun(@(x) imread(x), imagePaths, UniformOutput=false);
    imagesBatch = cellfun(@(x) imresize(x, opts.ResizeTo), images, UniformOutput=false);
    imagesBatch = cat(4, imagesBatch{:});
    imagesBatch = dlarray(single(imagesBatch), "SSCB");
end