% Classifing Neural Network Example
% author: Natanael Junior (natmourajr@gmail.com)
% LPS - Signal Processing Lab.
% UFRJ - Brazil

% Steps
% 1 - Data Aquisition
% 2 - Normalization (data, targets)
% 3 - Split Training Sets (train, test, validation)
% 4 - Training Process
% 5 - Result Analysis
% 
% In machine learning and related fields, artificial neural networks (ANNs)
% are computational models inspired by an animal's central nervous systems 
% (in particular the brain), and are used to estimate or approximate functions 
% that can depend on a large number of inputs and are generally unknown. 
% Artificial neural networks are generally presented as systems of interconnected 
% "neurons" which can compute values from inputs, and are capable of machine 
% learning as well as pattern recognition thanks to their adaptive nature.
% 
% This example intent to create a basic tutorial to classify 2 different
% class with MATLAB Neural Network Toolbox.

close all; % close all figure windows
clear all; % clear all variables
clc; % reset command line

fprintf('Starting %s.m\n',mfilename('fullpath'));
fprintf('Importing Functions\n');
addpath(genpath('functions'));

% 1 - Data Aquisition
load fisheriris; % load iris data base

% Iris Data base
% 
% 150 events (occurrences)
% 50 setosa class, 50 versicolor class, 50 virginca class
% meas (150x4):
% Attributes:
%  1. Sepal length in cm
%  2. Sepal width in cm
%  3. Petal length in cm
%  4. Petal width in cm
%
% species (150x1 - cell vector)
% string with class name


str_targets = species;

% 2 - Normalization (target)
fprintf('Normalizing Target\n');
targets =zeros(size(str_targets,1),1); 

for i = 1:size(str_targets,1)
    if strcmp(str_targets(i),'setosa')
        targets(i) = 1;
    end
    
    if strcmp(str_targets(i),'versicolor')
        targets(i) = 0;
    end
    
    if strcmp(str_targets(i),'virginica')
        targets(i) = -1;
    end
end

% selecting just two class
inputs = meas(find(targets == 1 | targets == 0),:);
targets = targets(find(targets == 1 | targets == -1));


% 3 - Split Training Sets (train, test, validation)
fprintf('Split Training Sets\n');
n_tests = 2;
CVO = cvpartition(length(targets),'Kfold',n_tests); % split into n_tests tests
trn_id =  CVO.training(1); % taking the first one
tst_id =  CVO.test(1); % taking the first one
val_id = tst_id; % test = validation -> small statistics


% turn trn_id, tst_id in integers  to use in NN training process
itrn = [];
itst = [];
ival = [];

for i = 1:length(trn_id)
    if trn_id(i) == 1
        itrn = [itrn; i];
    else 
        itst = [itst; i];
    end
end
ival = itst;

% 2 - Using train set to extract normalization factors
fprintf('Normalizing Inputs\n');
[~,ps] = mapstd(inputs(trn_id,:)'); % ps - normalization factors

% applying normalization in all events
% mapstd -> mean = 0, std = 1;
inputs_norm =  mapstd('apply',inputs',ps)';


% 4 - Training Process
fprintf('Training Process\n');

top = 2; % number of neurons in hidden layer
train_fnc = 'traingd'; % weights update function
perf_fnc = 'mse'; % error function
act_fnc = {'tansig'}; % activation function
n_epochs = 100;
show = true;

[trained_nn, train_description] = train_neural_network(inputs_norm', targets', itrn, ival, itst, top, train_fnc, perf_fnc, act_fnc, n_epochs, show);

nn_output = sim(trained_nn, inputs_norm');

% 5 - Result Analysis
fprintf('Result Analysis\n');

% train analysis
plotperform(train_description);
fig2pdf(gcf,'training_description.pdf');
set(gcf,'PaperOrientation','landscape');
saveas(gcf,'training_description.png'); close(gcf);

% separating 2 different class
c1 = nn_output((ival(find(targets(ival)==1)))); % validation set
c2 = nn_output((ival(find(targets(ival)==-1)))); % validation set

% find the maximum
max_value = max([(-1)*min(nn_output) max(nn_output)]);

% perform SP
[SP, pt_SPmax] = calc_sp(c1, c2, max_value);

% checking histograms
nn_hist_out(50,c1,c2,pt_SPmax);
fig2pdf(gcf,'histogram_4_class.pdf');
set(gcf,'PaperOrientation','landscape');
saveas(gcf,'histogram2class.png');
close(gcf);

% checking ROC
plot_roc(c1,c2,max_value);
fig2pdf(gcf,'roc.pdf');
set(gcf,'PaperOrientation','landscape');
saveas(gcf,'roc.png');
close(gcf);

% confusion matrix
plotconfusion((1+targets(ival)')/2, nn_output(ival));
fig2pdf(gcf,'confusion.pdf'); 
set(gcf,'PaperOrientation','landscape');
saveas(gcf,'confusion.png');
close(gcf);

fprintf('Exporting Functions\n');
rmpath(genpath('functions'));

fprintf('THE END!!!\n');


