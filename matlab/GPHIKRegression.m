% brief:    MATLAB class wrapper for the underlying Matlab-C++ Interface (GPHIKRegressionMex.cpp)
% author:   Alexander Freytag
% date:     17-01-2014 (dd-mm-yyyy)
classdef GPHIKRegression < handle
    
    properties (SetAccess = private, Hidden = true)
        % Handle to the underlying C++ class instance
        objectHandle; 
    end
    
    methods
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%      Constructor / Destructor    %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        %% constructor - create object
        function this = GPHIKRegression(varargin)
            this.objectHandle = GPHIKRegressionMex('new', varargin{:});
        end
        
        %% destructor - delete object
        function delete(this)
            GPHIKRegressionMex('delete', this.objectHandle);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%          Regression stuff        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         
        %% train - standard train - assumes initialized object
        function varargout = train(this, varargin)
            [varargout{1:nargout}] = GPHIKRegressionMex('train', this.objectHandle, varargin{:});
        end
        
        %% perform regression
        function varargout = estimate(this, varargin)
            [varargout{1:nargout}] = GPHIKRegressionMex('estimate', this.objectHandle, varargin{:});
        end 
        
        %% uncertainty - Uncertainty prediction
        function varargout = uncertainty(this, varargin)
            [varargout{1:nargout}] = GPHIKRegressionMex('uncertainty', this.objectHandle, varargin{:});
        end        

        %% test - evaluate regression on whole test set using L2 loss
        function varargout = testL2loss(this, varargin)
            [varargout{1:nargout}] = GPHIKRegressionMex('testL2loss', this.objectHandle, varargin{:});
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%       Online Learnable methods   %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% addExample
        function varargout = addExample(this, varargin)
            [varargout{1:nargout}] = GPHIKRegressionMex('addExample', this.objectHandle, varargin{:});
        end 
        %% addMultipleExamples
        function varargout = addMultipleExamples(this, varargin)
            [varargout{1:nargout}] = GPHIKRegressionMex('addMultipleExamples', this.objectHandle, varargin{:});
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%       Persistent methods         %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% store - store the classifier to an external file
        function varargout = store(this, varargin)
            [varargout{1:nargout}] = GPHIKRegressionMex('store', this.objectHandle, varargin{:});
        end
        %% restore -  load classifier from external file 
        function varargout = restore(this, varargin)
            [varargout{1:nargout}] = GPHIKRegressionMex('restore', this.objectHandle, varargin{:});
        end
    end
end