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

close all;
clear all;
clc;

fprintf('Starting %s.m\n',mfilename('fullpath'));
fprintf('Importing Functions\n');
addpath(genpath('functions'));

% 1 - Data Aquisition
load fisheriris;

str_targets = species;

% 2 - Normalization (target)
fprintf('Normalizing Target\n');
n_class = 3;
targets = zeros(size(str_targets,1),1); 
targets_norm = zeros(length(targets),n_class); 


for i = 1:size(str_targets,1)
    if strcmp(str_targets(i),'setosa')
        targets(i) = 1;
        targets_norm(i,:) = [1 -1 -1];
    end
    
    if strcmp(str_targets(i),'versicolor')
        targets(i) = 0;
        targets_norm(i,:) = [-1 1 -1];
    end
    
    if strcmp(str_targets(i),'virginica')
        targets(i) = -1;
        targets_norm(i,:) = [-1 -1 1];
    end
end

inputs = meas;

% 3 - Split Training Sets (train, test, validation)
fprintf('Split Training Sets\n');
n_tests = 2;
CVO = cvpartition(length(targets),'Kfold',n_tests); % split into n_tests tests
trn_id =  CVO.training(1); % taking the first one
tst_id =  CVO.test(1); % taking the first one
val_id = tst_id; % test = validation -> small statistics


% turn trn_id in integers
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

top = 10; % number of neurons in hidden layer
train_fnc = 'trainbfg'; % weights update function
perf_fnc = 'mse'; % error function
act_fnc = {'tansig'}; % activation function
n_epochs = 100;
show = true;

[trained_nn, train_description] = train_neural_network(inputs_norm', targets_norm', itrn, ival, itst, top, train_fnc, perf_fnc, act_fnc, n_epochs, show);

nn_output = sim(trained_nn, inputs_norm');

% 5 - Result Analysis
fprintf('Result Analysis\n');

% train analysis
plotperform(train_description);
fig2pdf(gcf,'training_description.pdf'); close(gcf);

% confusion matrix
plotconfusion(targets_norm(ival,:)', nn_output(:,ival));

l_aux = {'Setosa', 'Versicolor', 'Virginica' ,'Total'}; % labels
set(gca,'XTickLabel',l_aux);
set(gca,'YTickLabel',l_aux);
fig2pdf(gcf,'confusion.pdf'); close(gcf);

fprintf('Exporting Functions\n');
rmpath(genpath('functions'));

fprintf('THE END!!!\n');


