function fig2pdf(handler, title, paper_orient)
%FIG2PDF Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
    paper_orient = 'landscape';
end

set(handler,'units','centimeters');
set(handler,'PaperOrientation',paper_orient);
paper_coord = get(handler,'PaperSize');
set(handler,'PaperPosition',[0.1 0.1 paper_coord(1)-0.1 paper_coord(2)-0.1]);

saveas(handler,title);

end

