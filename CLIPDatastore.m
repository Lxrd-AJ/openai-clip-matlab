classdef CLIPDatastore < matlab.io.Datastore & matlab.io.datastore.Shuffleable
    % CLIPDatastore

    properties (Access=private)
        ImageCaptionMap
        ImageFolder
    end

    % Index tracking
    properties(Access=private)
        Index
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

        function shuffledThis = shuffle(this)
            % Create a copy of datastore
            shuffledThis = copy(this);

            shuffled = randperm(numel(this.IndexImages));
            shuffledThis.IndexImages = this.IndexImages(shuffled);
            reset(shuffledThis);
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
            out = caption;
            disp("TODO!")
        end
    end
end