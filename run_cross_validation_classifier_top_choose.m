% Classifing Neural Network Example
% author: Natanael Junior (natmourajr@gmail.com)
% LPS - Signal Processing Lab.
% UFRJ - Brazil

% Steps
% 1 - Data Aquisition
% 2 - Normalization (data, targets)
% 3 - Split Training Sets (train, test, validation)
% 4 - Training Process with Cross validation and Topology Sweep
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
n_tests = 10;
CVO = cvpartition(length(targets),'Kfold',n_tests); % split into n_tests tests

fprintf('Start Cross Validation Process with Topology Sweep\n');
min_top = 10;
max_top = 13;

s_top = min_top:max_top;

cv_SP = zeros(n_tests,length(s_top));
cv_Pd = zeros(n_tests,length(s_top));
cv_Pf = zeros(n_tests,length(s_top));

for i_sweep_top = 1:length(s_top)
    for i_cross_valid = 1:CVO.NumTestSets
        fprintf('Top %i (%i Neurons) - Set no %i\n', i_sweep_top, s_top(i_sweep_top), i_cross_valid);
        trn_id =  CVO.training(i_cross_valid); % taking the first one
        tst_id =  CVO.test(i_cross_valid); % taking the first one
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
        [~,ps] = mapstd(inputs(trn_id,:)'); % ps - normalization factors
        
        % applying normalization in all events
        % mapstd -> mean = 0, std = 1;
        inputs_norm =  mapstd('apply',inputs',ps)';
        
        
        % 4 - Training Process
        top = s_top(i_sweep_top); % number of neurons in hidden layer
        train_fnc = 'trainbfg'; % weights update function
        perf_fnc = 'mse'; % error function
        act_fnc = {'tansig'}; % activation function
        n_epochs = 100;
        show = false;
        
        [trained_nn, train_description] = train_neural_network(inputs_norm', targets', itrn, ival, itst, top, train_fnc, perf_fnc, act_fnc, n_epochs, show);
        
        nn_output = sim(trained_nn, inputs_norm');
        
        % 5 - Result Analysis
        % separating 2 different class
        c1 = nn_output((ival(find(targets(ival)==1)))); % validation set
        c2 = nn_output((ival(find(targets(ival)==-1)))); % validation set
        
        % find the maximum
        max_value = max([(-1)*min(nn_output) max(nn_output)]);
        
        % perform SP
        [SP, pt_SPmax] = calc_sp(c1, c2, max_value);
        
        Pd = length(find(c1 > pt_SPmax))/length(c1); % Detection Probability
        Pf = length(find(c2 > pt_SPmax))/length(c2); % False-Alarm Probability
        
        cv_SP(i_cross_valid,i_sweep_top) = SP;
        cv_Pd(i_cross_valid,i_sweep_top) = Pd;
        cv_Pf(i_cross_valid,i_sweep_top) = Pf;
    end
end

% checking all SP, Pd, Pf - for all sets
figure;
plot(1:CVO.NumTestSets, mean(cv_SP,2),'bx');
hold on;
plot(1:CVO.NumTestSets, mean(cv_Pd,2),'ro');
plot(1:CVO.NumTestSets, mean(cv_Pf,2),'gs');
hold off;
ylabel('% SP / % Pd / % Pf','FontSize', 15,'FontWeight', 'bold');
xlabel('Set Id','FontSize', 15,'FontWeight', 'bold');
title(sprintf('Cross Validation Results - %i Sets',CVO.NumTestSets),'FontSize', 15,'FontWeight', 'bold');
legend('% SP', '% Pd', '% Pf');

fig2pdf(gcf,'cv_result.pdf'); close(gcf);

% checking all SP, Pd, Pf - for all sets
figure;
plot(min_top:max_top, mean(cv_SP,1),'bx');
hold on;
plot(min_top:max_top, mean(cv_Pd,1),'ro');
plot(min_top:max_top, mean(cv_Pf,1),'gs');
hold off;
ylabel('% SP / % Pd / % Pf','FontSize', 15,'FontWeight', 'bold');
xlabel('# Neurons','FontSize', 15,'FontWeight', 'bold');
title(sprintf('Sweep Topologies - #%i Tops',length(s_top)),'FontSize', 15,'FontWeight', 'bold');
legend('% SP', '% Pd', '% Pf');

fig2pdf(gcf,'top_sweep_result.pdf'); close(gcf);


fprintf('Exporting Functions\n');
rmpath(genpath('functions'));

fprintf('THE END!!!\n');


