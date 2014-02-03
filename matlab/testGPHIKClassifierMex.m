% brief:    Demo-program showing how to use the GPHIK Interface (without a class wrapper)
% author:   Alexander Freytag
% date:     07-01-2014 (dd-mm-yyyy)

myData = [ 0.2 0.3 0.5;
           0.3 0.2 0.5;
           0.9 0.0 0.1;
           0.8 0.1 0.1;
           0.1 0.1 0.8;
           0.1 0.0 0.9
          ];
myLabels = [1,1,2,2,3,3];


% init new GPHIKClassifier object
myGPHIKClassifier = GPHIKClassifierMex ( 'new', 'verbose', 'false', ...
    'optimization_method', 'none', 'varianceApproximation', 'approximate_rough',...
    'nrOfEigenvaluesToConsiderForVarApprox',4,...
    'uncertaintyPredictionForClassification', false ...
    );

% run train method
GPHIKClassifierMex ( 'train', myGPHIKClassifier, myData, myLabels);

myDataTest = [ 0.3 0.4 0.3
             ];
myLabelsTest = [1];

% run single classification call
[ classNoEst, score, uncertainty] = GPHIKClassifierMex ( 'classify', myGPHIKClassifier, myDataTest )
% compute predictive variance
uncertainty = GPHIKClassifierMex ( 'uncertainty', myGPHIKClassifier, myDataTest )
% run test method evaluating arr potentially using multiple examples
[ arr, confMat, scores] = GPHIKClassifierMex ( 'test', myGPHIKClassifier, myDataTest, myLabelsTest )

% add a single new example
newExample = [ 0.5 0.5 0.0
             ];
newLabel = [4];
GPHIKClassifierMex ( 'addExample', myGPHIKClassifier, newExample, newLabel);

% add mutiple new examples
newExamples = [ 0.3 0.3 0.4;
                0.1, 0.2, 0.7
             ];
newLabels = [1,3];
GPHIKClassifierMex ( 'addMultipleExamples', myGPHIKClassifier, newExamples, newLabels );

% perform evaluation again

% run single classification call
[ classNoEst, score, uncertainty] = GPHIKClassifierMex ( 'classify', myGPHIKClassifier, myDataTest )
% compute predictive variance
uncertainty = GPHIKClassifierMex ( 'uncertainty', myGPHIKClassifier, myDataTest )
% run test method evaluating arr potentially using multiple examples
[ arr, confMat, scores] = GPHIKClassifierMex ( 'test', myGPHIKClassifier, myDataTest, myLabelsTest )

% clean up and delete object
GPHIKClassifierMex ( 'delete',myGPHIKClassifier);