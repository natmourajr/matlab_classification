function nn_hist_out(nbins, class1, class2, decision_ths)
%NN_HIST_OUT Summary of this function goes here
%   Detailed explanation goes here


figure;
x_min = 0;
x_max = 0;


x_min = min([class1 class2]);
x_max = max([class1 class2]);
[class1_x, class1_y] = hist(class1,nbins);
stairs(class1_y,class1_x,'b');
hold on;
[class2_x, class2_y] = hist(class2,nbins);
stairs(class2_y,class2_x,'r');
a = axis;
a(1) = x_min;
a(2) = x_max;
axis(a);

aux = a(3):0.01:a(4);
size(decision_ths);
plot(decision_ths*ones(size(aux,2),1),aux,'k--');


legend('Class 1','Class 2','Decision Trh.');

xlabel('Value');
ylabel('Occurrence');
hold off;

end

