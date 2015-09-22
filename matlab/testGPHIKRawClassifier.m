% brief:    Demo-program showing how to use the GPHIKRaw Interface (without a class wrapper)
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
myGPHIKRawClassifier = GPHIKRawClassifierMex ( ...
     'new', ...
     'verbose', 'false' ...
    );

% run train method
GPHIKRawClassifierMex ( 'train', myGPHIKRawClassifier, myData, myLabels);

myDataTest = [ 0.3 0.4 0.3
             ];
myLabelsTest = [1];

% run single classification call
[ classNoEst, score ]   = GPHIKRawClassifierMex ( 'classify', myGPHIKRawClassifier, myDataTest )

% run test call which classifies entire data set
[ arr, confMat, scores] = GPHIKRawClassifierMex ( 'test', myGPHIKRawClassifier, myDataTest, myLabelsTest )



% clean up and delete object
GPHIKRawClassifierMex ( 'delete',myGPHIKRawClassifier);