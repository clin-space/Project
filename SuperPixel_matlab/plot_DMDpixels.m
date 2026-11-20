function plot_DMDpixels(DMDpixels,name)

[ny,nx] = size(DMDpixels);

figure; imagesc(abs(DMDpixels));
set(gca,'YTick',[1 ny/2 ny],'XTick',[1 nx/2 nx],'FontSize',16)
title(['DMD pattern ' name],'fontsize',24)
h=colorbar;
colormap(flipud(gray))
set(h,'YTickLabel',{'off','on'},'YTick',0:1,'FontSize',16)
axis image