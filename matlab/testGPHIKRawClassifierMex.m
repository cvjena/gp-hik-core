% brief:    Demo-program showing how to use the GPHIKRawClassifier
%           Interface ( with a class wrapper)
% author:   Alexander Freytag
% date:     21-09-2015 (dd-mm-yyyy)

myData = [ 0.2 0.3 0.5;
           0.3 0.2 0.5;
           0.9 0.0 0.1;
           0.8 0.1 0.1;
           0.1 0.1 0.8;
           0.1 0.0 0.9
          ];
myLabels = [1,1,2,2,3,3];


% init new GPHIKRawClassifier object
myGPHIKRawClassifier = GPHIKRawClassifier ( ...
                       'verbose', 'false' ...
                     );

% run train method
myGPHIKRawClassifier.train( myData, myLabels );

myDataTest = [ 0.3 0.4 0.3
             ];
myLabelsTest = [1];

% run single classification call
[ classNoEst, score] = myGPHIKRawClassifier.classify ( myDataTest )
% run test method evaluating arr potentially using multiple examples
[ arr, confMat, scores] = myGPHIKRawClassifier.test( myDataTest, myLabelsTest )


% clean up and delete object
myGPHIKRawClassifier.delete();
clear ( 'myGPHIKRawClassifier' );