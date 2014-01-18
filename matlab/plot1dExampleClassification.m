myData = [ 0.2; 0.8];
% create l1-normalized 'histograms'
myData = cat(2,myData , 1-myData)';
myLabels = [1,2];


% init new GPHIKClassifier object
myGPHIKClassifier = GPHIKClassifier ( 'verbose', 'false', ...
    'optimization_method', 'none', 'varianceApproximation', 'approximate_fine',...
    'nrOfEigenvaluesToConsiderForVarApprox',2,...
    'uncertaintyPredictionForClassification', true ...
    );

% run train method
myGPHIKClassifier.train( myData, myLabels );

myDataTest = 0:0.01:1;
% create l1-normalized 'histograms'
myDataTest = cat(1, myDataTest, 1-myDataTest)';


scores = zeros(size(myDataTest,1),1);
uncertainties = zeros(size(myDataTest,1),1);
for i=1:size(myDataTest,1)
    example = myDataTest(i,:);
    [ classNoEst, score, uncertainties(i)] = myGPHIKClassifier.classify( example );
    scores(i) = score(1);
end



figure;
hold on;

%#initialize x array
x=0:0.01:1;

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

plot ( x,scores,'rx');


% clean up and delete object
myGPHIKClassifier.delete();

clear ( 'myGPHIKClassifier' );