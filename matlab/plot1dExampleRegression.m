% BRIEF: Small visualization script using the GPHIKRegression
% author: Alexander Freytag
% date: 20-01-2014 (dd-mm-yyyy)

myData = [ 0.1; 0.3; 0.7; 0.8];
% create l1-normalized 'histograms'
myData = cat(2,myData , 1-myData);
myValues = [0.3; 0.0; 1.0; 1.4];


% init new GPHIKRegression object
myGPHIKRegression = GPHIKRegression ( 'verbose', 'false', ...
    'optimization_method', 'none', ...
    'varianceApproximation', 'approximate_fine',...
    'nrOfEigenvaluesToConsiderForVarApprox',1,...
    'uncertaintyPredictionForRegression', true, ...
    'noise', 0.000001 ...
    );

    %'varianceApproximation', 'approximate_fine',...
    %'varianceApproximation', 'exact',...

% run train method
myGPHIKRegression.train( myData, myValues );

myDataTest = 0:0.01:1;
% create l1-normalized 'histograms'
myDataTest = cat(1, myDataTest, 1-myDataTest)';


scores = zeros(size(myDataTest,1),1);
uncertainties = zeros(size(myDataTest,1),1);
for i=1:size(myDataTest,1)
    example = myDataTest(i,:);
    [ scores(i), uncertainties(i)] = myGPHIKRegression.estimate( example );
end


% create figure and set title
classificationFig = figure;
set ( classificationFig, 'name', 'Regression with GPHIK');

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
%#plot filled area for predictive variance ( aka regression uncertainty )
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
plot ( myData(:,1), myValues, ...
       'LineStyle', 'none', ...
       'LineWidth', 3, ...
       'Marker','o', ...
       'MarkerSize',6, ...
       'MarkerEdgeColor','b', ...
       'MarkerFaceColor',[0.5,0.5,0.5] ...
       );

xlabel('1st Input dimension');
ylabel('Regression score');   

i_fontSizeAxis = 16;
set(get(gca,'XLabel'), 'FontSize', i_fontSizeAxis);
set(get(gca,'YLabel'), 'FontSize', i_fontSizeAxis);

   

% clean up and delete object
myGPHIKRegression.delete();

clear ( 'myGPHIKRegression' );