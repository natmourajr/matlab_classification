function [ max_sp, pt_max_sp] = calc_sp( signal, noise, lim)
%CALC_SP Summary of this function goes here
%   Detailed explanation goes here

if(size(signal,1) > size(signal,2))
    signal = signal';
end

if(size(noise,1) > size(noise,2))
    noise = noise';
end

n_pts = 100;
vmin = (-1)*lim;
vmax = lim; 
ths = linspace(vmin,vmax,n_pts);

pf = zeros(1,length(ths));
pd = zeros(1,length(ths));
%SP = zeros(1,length(ths));

for i = 1:length(ths)
    th = ths(i);
    pd(i) = length(find(signal > th))/length(signal);
    pf(i) = length(find(noise > th))/length(noise);
    SP(i) = sqrt(sqrt((pd(i)*(1-pf(i))))*(pd(i) + (1-pf(i)))/2);
end

max_sp = max(SP);

pt_max_sp = ths(find(SP == max_sp));
pt_max_sp = pt_max_sp(1); % get the first one;


end

