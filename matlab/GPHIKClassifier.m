% brief:    MATLAB class wrapper for the underlying Matlab-C++ Interface (GPHIK.cpp)
% author:   Alexander Freytag
% date:     07-01-2014 (dd-mm-yyyy)
classdef GPHIKClassifier < handle
    
    properties (SetAccess = private, Hidden = true)
        % Handle to the underlying C++ class instance
        objectHandle; 
    end
    
    methods
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%      Constructor / Destructor    %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        %% constructor - create object
        function this = GPHIKClassifier(varargin)
            this.objectHandle = GPHIK('new', varargin{:});
        end
        
        %% destructor - delete object
        function delete(this)
            GPHIK('delete', this.objectHandle);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%       Classification stuff       %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         
        %% train - standard train - assumes initialized object
        function varargout = train(this, varargin)
            [varargout{1:nargout}] = GPHIK('train', this.objectHandle, varargin{:});
        end
        
        %% classify
        function varargout = classify(this, varargin)
            [varargout{1:nargout}] = GPHIK('classify', this.objectHandle, varargin{:});
        end 
        
        %% uncertainty - Uncertainty prediction
        function varargout = uncertainty(this, varargin)
            [varargout{1:nargout}] = GPHIK('uncertainty', this.objectHandle, varargin{:});
        end        

        %% test - evaluate classifier on whole test set
        function varargout = test(this, varargin)
            [varargout{1:nargout}] = GPHIK('test', this.objectHandle, varargin{:});
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%       Online Learnable methods   %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% addExample
        function varargout = addExample(this, varargin)
            [varargout{1:nargout}] = GPHIK('addExample', this.objectHandle, varargin{:});
        end 
        %% addMultipleExamples
        function varargout = addMultipleExamples(this, varargin)
            [varargout{1:nargout}] = GPHIK('addMultipleExamples', this.objectHandle, varargin{:});
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%       Persistent methods         %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% store - store the classifier to an external file
        function varargout = store(this, varargin)
            [varargout{1:nargout}] = GPHIK('store', this.objectHandle, varargin{:});
        end
        %% restore -  load classifier from external file 
        function varargout = restore(this, varargin)
            [varargout{1:nargout}] = GPHIK('restore', this.objectHandle, varargin{:});
        end
    end
end