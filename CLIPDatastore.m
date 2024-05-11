classdef CLIPDatastore < matlab.io.Datastore & matlab.io.datastore.Shuffleable
    % CLIPDatastore

    properties (Access=private)
        ImageCaptionMap
        ImageFolder
        BertTokenizer
    end

    % Index tracking
    properties(Access=private)
        Index
        % A list of all the image file names read from the token .txt (flickr-dataset/Flickr8k.lemma.token.txt)
        % used to maintain an index of images that have been read
        IndexImages
    end

    methods
        function this = CLIPDatastore(opts)
            arguments
                opts.Lemma = "./Flickr8k_text/Flickr8k.lemma.token.txt"
                opts.ImageFolder = "./Flicker8k_Dataset"
            end
            this.ImageFolder = opts.ImageFolder;

            captions = readlines(opts.Lemma);
            captions = arrayfun(@(x) strsplit(x, "\t"), captions, UniformOutput=false);
            captions = vertcat(captions{:});
            
            keys = captions(:,1);
            values = captions(:,2);
            % Store as in-memory dictionary
            this.ImageCaptionMap = dictionary(keys, values);
            this.IndexImages = keys;
            this.Index = 1;

            [~, this.BertTokenizer] = bert();

            reset(this);
        end

        function tf = hasdata(this)
            tf = this.Index <= numel(this.IndexImages);
        end

        function [image, tokenisedCaption, caption] = read(this)
            if ~hasdata(this)
                error("No more data to read");
            end

            keyToUse = this.IndexImages(this.Index);
            caption = this.ImageCaptionMap(keyToUse);

            % Prepare the image
            keyParts = split(keyToUse, "#");
            imageName = keyParts(1);
            imagePath = fullfile(this.ImageFolder, imageName);
            image = imread(imagePath);

            % Tokenise the caption
            tokenisedCaption = this.tokenize(caption);

            this.Index = this.Index + 1;
        end

        function reset(this)
            this.Index = 1;
        end

        function shuffledThis = shuffle(this, opts)
            arguments
                this
                opts.PercentageToKeep = 1
            end
            % Create a copy of datastore
            shuffledThis = copy(this);

            shuffled = randperm(numel(this.IndexImages));

            endIndex = floor(opts.PercentageToKeep * numel(this.IndexImages));
            shuffled = shuffled(1:endIndex);
            shuffledThis.IndexImages = this.IndexImages(shuffled);
            
            reset(shuffledThis);
        end

        function count = numel(this)
            count = numel(this.IndexImages);
        end
    end

    methods (Hidden = true)
        function frac = progress(this)
            % Determine percentage of data read from datastore
            if hasdata(this) 
                frac = (this.Index-1)/numel(this.IndexImages);
            else 
                frac = 1;
            end 
        end
    end

    methods(Access=private)
        function out = tokenize(this, caption)
            tokenized = encode(this.BertTokenizer, caption);
            out = tokenized{1};
        end
    end
end