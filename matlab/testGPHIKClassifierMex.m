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
myGPHIKClassifier = GPHIKClassifier ( ...
    'verbose', 'false', ...
    'optimization_method', 'none', ...
    'varianceApproximation', 'approximate_rough',...
    'nrOfEigenvaluesToConsiderForVarApprox',4,...
    'uncertaintyPredictionForClassification', false ...
    );

% run train method
myGPHIKClassifier.train( myData, myLabels );

myDataTest = [ 0.3 0.4 0.3
             ];
myLabelsTest = [1];

% run single classification call
[ classNoEst, score, uncertainty] = myGPHIKClassifier.classify( myDataTest )
% compute predictive variance
uncertainty = myGPHIKClassifier.uncertainty( myDataTest )
% run test method evaluating arr potentially using multiple examples
[ arr, confMat, scores] = myGPHIKClassifier.test( myDataTest, myLabelsTest )

% add a single new example
newExample = [ 0.5 0.5 0.0
             ];
newLabel = [4];
myGPHIKClassifier.addExample( newExample, newLabel );

% add mutiple new examples
newExamples = [ 0.3 0.3 0.4;
                0.1, 0.2, 0.7
             ];
newLabels = [1,3];
myGPHIKClassifier.addMultipleExamples( newExamples, newLabels );

% perform evaluation again

% run single classification call
[ classNoEst, score, uncertainty] = myGPHIKClassifier.classify( myDataTest )
% compute predictive variance
uncertainty = myGPHIKClassifier.uncertainty( myDataTest )
% run test method evaluating arr potentially using multiple examples
[ arr, confMat, scores] = myGPHIKClassifier.test( myDataTest, myLabelsTest )

% clean up and delete object
myGPHIKClassifier.delete();
clear ( 'myGPHIKClassifier' );