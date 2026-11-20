
function [E_target,maxIntensity] = makeFieldFromImage()

% Take two images, one for intensity and one for phase, and combine them
% into a complex field:

I = imread('images/pebbles_768.jpg');
I = (double(I(:,:,1))+double(I(:,:,2))+double(I(:,:,3)))/3;
I = I / 255;
I = min(1,I*1.3);
maxIntensity = max(max(I))/sum(sum(I));

phase = imread('images/barsik_768.jpg');
phase = double(phase(:,:,1));

E_target = sqrt(I).*exp((1.9*pi*phase/max(max(phase))-0.95*pi)*1i);


    
