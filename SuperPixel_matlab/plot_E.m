
function plot_E(E,maxIntensity,phaseColorMap,name)

[ny,nx] = size(E);

if strcmp(phaseColorMap, 'cmap')
    load('cmap.mat');
    phaseColorMap = cmap;
end

figure('units','normalized','position',[.1 .1 .7 .5])
subplot(1,2,1)
imagesc(abs(E).^2);
set(gca,'YTick',[1 ny/2 ny],'XTick',[1 nx/2 nx],'FontSize',16)
title([name ' intensity'],'fontsize',24)
h=colorbar;
colormap('gray')
caxis([0 maxIntensity])
set(h,'FontSize',16)
axis image

subplot(1,2,2)
imagesc(angle(E));
set(gca,'YTick',[1 ny/2 ny],'XTick',[1 nx/2 nx],'FontSize',16)
title([name ' phase'],'fontsize',24)
h=colorbar;
colormap(phaseColorMap)
caxis([-pi pi])
set(h,'FontSize',16)
axis image
