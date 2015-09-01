% brief:    Demo-program showing how to use the GPHIKClassifier Interface (including the class wrapper)
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


    % boolean
    b_verboseTime                       = false;
    b_verbose                           = false;   
    b_uncertaintyPredictionForClassification ...
                                        = false;
    b_optimize_noise                    = false;
    b_use_quantization                  = false;
    b_ils_verbose                       = false;
    %
    % integer
    i_nrOfEigenvaluesToConsiderForVarApprox ...
                                        = 4;
    i_num_bins                          = 100; % default
    i_ils_max_iterations                = 1000; % default
    %
    % double    
    d_ils_min_delta                     = 1e-7; % default
    d_ils_min_residual                  = 1e-7; % default
    d_noise                             = 0.1;  % default
    %
    % string    
    s_ils_method                        = 'CG'; % default
    s_optimization_method               = 'greedy';
    settings.settingsGPHIK.s_transform  = 'identity';
    settings.settingsGPHIK.s_varianceApproximation...
                                        = 'approximate_fine'; 


% init new GPHIKClassifier object
myGPHIKClassifier = ...
        GPHIKClassifier ( ...
                          'verboseTime',                               b_verboseTime, ...
                          'verbose',                                   b_verbose, ...
                          'uncertaintyPredictionForClassification',    b_uncertaintyPredictionForClassification, ...
                          'optimize_noise',                            b_optimize_noise, ...
                          'use_quantization',                          b_use_quantization, ...
                          'ils_verbose',                               b_ils_verbose, ...
                          ...                         
                          'num_bins',                                  i_num_bins, ...                           
                          'ils_max_iterations',                        i_ils_max_iterations, ...                          
                          ...
                          'ils_min_delta',                             d_ils_min_delta, ...
                          'ils_min_residual',                          d_ils_min_residual, ...
                          'noise',                                     d_noise, ...                          
                          ...
                          'ils_method',                                s_ils_method, ...
                          'optimization_method',                       s_optimization_method ...
        );    
%     GPHIKClassifier (...
%     'verbose',                           'false', ...
%     'optimization_method',               'none', ...
%     'varianceApproximation', 'approximate_fine',...
%     'nrOfEigenvaluesToConsiderForVarApprox', 4,...
%     'uncertaintyPredictionForClassification', false ...
%     
%     );

% run train method
myGPHIKClassifier.train( myData, myLabels );

% check the reclassification is working!
[ arrReCl, confMatReCl, scoresReCl] = myGPHIKClassifier.test( myData, myLabels )
uncertainty = myGPHIKClassifier.uncertainty( myData(1,:) )

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
myGPHIKClassifier.addExample( newExample, newLabel);

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