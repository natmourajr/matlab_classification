function plot_roc( signal, noise, lim)
%PLOT_ROC Summary of this function goes here
%   Detailed explanation goes here

if(size(signal,1) > size(signal,2))
    signal = signal';
end
if(size(noise,1) > size(noise,2))
    noise = noise';
end

dist = 100;
vmin = (-1)*lim;
vmax = lim; 
ths = linspace(vmin,vmax,dist);

pf = zeros(1,length(ths));
pd = zeros(1,length(ths));

for i = 1:length(ths)
    th = ths(i);
    pd(i) = length(find(signal > th))/length(signal);
    pf(i) = length(find(noise > th))/length(noise);
    SP(i) = sqrt(sqrt((pd(i)*(1-pf(i))))*(pd(i) + (1-pf(i)))/2);
end
figure;
color = 'b';
hline = plot(100*pf,100*pd,'linewidth',2,'Color',color);
sp_max = find(SP== max(SP));
text(100*pf(sp_max(1))+3,100*pd(sp_max(1))-7,sprintf('SP: %1.2f%%\nP_{D}: %1.2f%%\nP_{FA}: %1.2f%%',100*max(SP),100*pd(sp_max(1)),100*pf(sp_max(1))));
hold on
plot(100*pf(sp_max(1)),100*pd(sp_max(1)),'ro','Color','r','MarkerSize',10,'LineWidth',3);
hold off
axis([0 100 0 100]);
grid on;

set(gca,'XTick',[0:10:100]);
title('Receiver Operating Characteristic');
ylabel('Detection Efficence(%)');
xlabel('False Alarm (%)');

end

