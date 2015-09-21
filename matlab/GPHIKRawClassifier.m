% brief:    MATLAB class wrapper for the underlying Matlab-C++ Interface (GPHIKRawClassifierMex.cpp)
% author:   Alexander Freytag
% date:     07-01-2014 (dd-mm-yyyy)
classdef GPHIKRawClassifier < handle

    properties (SetAccess = private, Hidden = true)
        % Handle to the underlying C++ class instance
        objectHandle;
    end

    methods

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%      Constructor / Destructor    %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% constructor - create object
        function this = GPHIKRawClassifier(varargin)
            this.objectHandle = GPHIKRawClassifierMex('new', varargin{:});
        end

        %% destructor - delete object
        function delete(this)
            GPHIKRawClassifierMex('delete', this.objectHandle);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%       Classification stuff       %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% train - standard train - assumes initialized object
        function varargout = train(this, varargin)
            [varargout{1:nargout}] = GPHIKRawClassifierMex('train', this.objectHandle, varargin{:});
        end

        %% classify
        function varargout = classify(this, varargin)
            [varargout{1:nargout}] = GPHIKRawClassifierMex('classify', this.objectHandle, varargin{:});
        end


        %% test - evaluate classifier on whole test set
        function varargout = test(this, varargin)
            [varargout{1:nargout}] = GPHIKRawClassifierMex('test', this.objectHandle, varargin{:});
        end

    end
end
