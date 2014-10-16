% Classifing Neural Network Example
% author: Natanael Junior (natmourajr@gmail.com)
% LPS - Signal Processing Lab.
% UFRJ - Brazil

% Steps
% 1 - Data Aquisition
% 2 - Normalization (data, targets)
% 3 - Split Training Sets (train, test, validation)
% 4 - Training Process with Cross Validation
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

cv_SP = zeros(n_tests,1);
cv_Pd = zeros(n_tests,1);
cv_Pf = zeros(n_tests,1);

warning 'off'
for i_cross_validation = 1:CVO.NumTestSets
    fprintf('Set no %i\n', i_cross_valid);
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
    show = false;
    
    [trained_nn, train_description] = train_neural_network(inputs_norm', targets_norm', itrn, ival, itst, top, train_fnc, perf_fnc, act_fnc, n_epochs, show);
    
    nn_output = sim(trained_nn, inputs_norm');
    
    % 5 - Result Analysis
    fprintf('Result Analysis\n');
    
    % confusion matrix
    [a,b,c,d] = confusion(targets_norm(ival,:)', nn_output(:,ival));
    
    id_rm_class = [];
    
    % Perform SP calc.
    for i_class = 1:n_class
        class_count = 0;
        for i_event = 1:length(ival)
            if targets_norm(ival(i_event),i_class) == 1
                class_count = 1;
            end
            if class_count == 1
                break;
            end
        end
        if class_count == 0
            id_rm_class = [id_rm_class i_class];
        end
    end
    
    SP = 0;
    arit_mean = 0;
    geo_mean  = 1;
    if isempty(id_rm_class)
        for i_class = 1:n_class
            arit_mean = arit_mean+d(i_class,3);
            geo_mean = geo_mean*d(i_class,3);
        end
        arit_mean = arit_mean/n_class;
        geo_mean = geo_mean^(1/n_class);        
    else
        for i_class = 1:n_class
            if ismember(i_class,id_rm_class)
                continue;
            end
            arit_mean = arit_mean+d(i_class,3);
            geo_mean = geo_mean*d(i_class,3);
        end
        arit_mean = arit_mean/(n_class-length(id_rm_class));
        geo_mean = geo_mean^(1/(n_class-length(id_rm_class)));
    end
    
    SP = sqrt(arit_mean*geo_mean);
    
    cv_SP(i_cross_validation) = SP;
    
    
end
warning 'on'

% checking all SP
figure;
plot(1:CVO.NumTestSets, cv_SP,'bx','LineWidth',2.0);
ylabel('% SP','FontSize', 15,'FontWeight', 'bold');
xlabel('Set Id','FontSize', 15,'FontWeight', 'bold');
title(sprintf('Cross Validation: SP Results - %i Sets',CVO.NumTestSets),'FontSize', 15,'FontWeight', 'bold');

fig2pdf(gcf,'cv_result.pdf'); close(gcf);


fprintf('Exporting Functions\n');
rmpath(genpath('functions'));

fprintf('THE END!!!\n');


