classdef CLIPDatastore < matlab.io.Datastore
    % CLIPDatastore

    properties (Access=private)
        Dataset
        ImageFolder
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
            
            % Store as in-memory dictionary
            keys = captions(:,1);
            values = captions(:,2);
            
            this.Dataset = dictionary(keys, values);
        end
    end
end