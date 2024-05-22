classdef CLIP < handle
    % CLIP

    properties(Access=private)
        Net
        ImageInputSize
        Temperature
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
    end
end