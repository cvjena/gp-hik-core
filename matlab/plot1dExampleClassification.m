% BRIEF: Small visualization script using GPHIKClassifier
% author: Alexander Freytag
% date: 20-01-2014 (dd-mm-yyyy)

myData = [ 0.2; 0.6; 0.9];
% create l1-normalized 'histograms'
myData = cat(2,myData , 1-myData);
myLabels = [1; 2; 2];


% init new GPHIKClassifier object
myGPHIKClassifier = GPHIKClassifier ( 'verbose', 'false', ...
    'optimization_method', 'none', 'varianceApproximation', 'approximate_fine',...
    'nrOfEigenvaluesToConsiderForVarApprox',2,...
    'uncertaintyPredictionForClassification', true, ...
    'noise', 0.000001 ...
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


% create figure and set title
classificationFig = figure;
set ( classificationFig, 'name', 'Classification with GPHIK');

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


% clean up and delete object
myGPHIKClassifier.delete();

clear ( 'myGPHIKClassifier' );