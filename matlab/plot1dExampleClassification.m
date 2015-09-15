% BRIEF: Small visualization script using GPHIKClassifier
% author: Alexander Freytag
% date: 20-01-2014 (dd-mm-yyyy)

myData = [ 0.2; 0.6; 0.9];
% create l1-normalized 'histograms'
myData = cat(2,myData , 1-myData);
myLabels = [1; 2; 2];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% boolean
%interested in time measurements?
b_verboseTime                       = false;

%interested in outputs?
b_verbose                           = false;  
b_debug                             = false;  


% important for plotting!
b_uncertaintyPredictionForClassification ...
                                    = true; 
b_optimize_noise                    = false;
b_use_quantization                  = true;
b_ils_verbose                       = false;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% integer
i_nrOfEigenvaluesToConsiderForVarApprox ...
                                    = 1;
i_num_bins                          = 1000; % default
i_ils_max_iterations                = 1000; % default
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% double    
d_ils_min_delta                     = 1e-7; % default
d_ils_min_residual                  = 1e-7; % default

% model regularization
d_noise                             = 0.000001; 

% adapt parameter bounds if you are interested in optimization
d_parameter_lower_bound             = 1.0;
d_parameter_upper_bound             = 1.0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% string    
s_ils_method                        = 'CG'; % default

% options: 'none', 'greedy', 'downhillsimplex'
s_optimization_method               = 'downhillsimplex';

% options: '1d-aequi-0-1' , '1d-aequi-0-max' or 'nd-aequi-0-1'
s_quantType                         = '1d-aequi-0-1';

% options:  'identity', 'exp', 'absexp'
% with settings above, this equals 'identity'
s_transform                         = 'identity'; 

% options: 'exact', 'approximate_fine', 'approximate_rough', and 'none'
s_varianceApproximation             = 'approximate_fine'; 

% init new GPHIKClassifier object
myGPHIKClassifier = ...
        GPHIKClassifier ( ...
                          'verboseTime',                               b_verboseTime, ...
                          'verbose',                                   b_verbose, ...
                          'debug',                                     b_debug, ...                          
                          'uncertaintyPredictionForClassification',    b_uncertaintyPredictionForClassification, ...
                          'optimize_noise',                            b_optimize_noise, ...
                          'use_quantization',                          b_use_quantization, ...
                          'ils_verbose',                               b_ils_verbose, ...
                          ...        
                          'nrOfEigenvaluesToConsiderForVarApprox',     i_nrOfEigenvaluesToConsiderForVarApprox, ...                          
                          'num_bins',                                  i_num_bins, ...                           
                          'ils_max_iterations',                        i_ils_max_iterations, ...                          
                          ...
                          'ils_min_delta',                             d_ils_min_delta, ...
                          'ils_min_residual',                          d_ils_min_residual, ...
                          'noise',                                     d_noise, ...        
                          'parameter_lower_bound',                     d_parameter_lower_bound, ...
                          'parameter_upper_bound',                     d_parameter_upper_bound, ...
                          ...
                          'ils_method',                                s_ils_method, ...
                          'optimization_method',                       s_optimization_method, ...   
                          's_quantType',                               s_quantType, ...                          
                          'transform',                                 s_transform, ...
                          'varianceApproximation',                     s_varianceApproximation ...
        );

%% run train method
myGPHIKClassifier.train( myData, myLabels );


%% evaluate model on test data

myDataTest = 0:0.001:1;
% create l1-normalized 'histograms'
myDataTest = cat(1, myDataTest, 1-myDataTest)';


scores = zeros(size(myDataTest,1),1);
uncertainties = zeros(size(myDataTest,1),1);
for i=1:size(myDataTest,1)
    example = myDataTest(i,:);

%     [ classNoEst, score, uncertainties(i)] = myGPHIKClassifier.classify( sparse(example) );
    [ classNoEst, score, uncertainties(i)] = myGPHIKClassifier.classify( example );    
%      [ classNoEst, score] = myGPHIKClassifier.classify( example );
%     [ classNoEst, score] = myGPHIKClassifier.classify( sparse(  example ) );    
    scores(i) = score(1);
end

%% plot results

% create figure and set title
classificationFig = figure;
set ( classificationFig, 'name', 'Classification with GPHIK');

hold on;

%#initialize x array
x=myDataTest(:,1)';

%#create first curve
uncLower=scores-uncertainties;
%#create second curve
uncUpper=scores+uncertainties;


%#create polygon-like x values for plotting
X=[x,fliplr(x)];
%# concatenate y-values accordingly
Y=[uncLower',fliplr(uncUpper')]; 
%#plot filled area
fill(X,Y,'y');                  

% plot mean values
plot ( x,scores, ...
       'LineStyle', '--', ...
       'LineWidth', 2, ...
       'Color', 'r', ...
       'Marker','none', ...
       'MarkerSize',1, ...
       'MarkerEdgeColor','r', ...
       'MarkerFaceColor',[0.5,0.5,0.5] ...
       );

% plot training data
plot ( myData(:,1), 2*(myLabels==1)-1, ...
       'LineStyle', 'none', ...
       'LineWidth', 3, ...
       'Marker','o', ...
       'MarkerSize',6, ...
       'MarkerEdgeColor','b', ...
       'MarkerFaceColor',[0.5,0.5,0.5] ...
       );

xlabel('1st Input dimension');
ylabel('Classification score');   

i_fontSizeAxis = 16;
set(get(gca,'XLabel'), 'FontSize', i_fontSizeAxis);
set(get(gca,'YLabel'), 'FontSize', i_fontSizeAxis);


%% clean up and delete object
myGPHIKClassifier.delete();

clear ( 'myGPHIKClassifier' );