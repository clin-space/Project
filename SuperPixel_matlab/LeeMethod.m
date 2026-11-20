
function [DMDpattern,E_obtained,fidelity,efficiency] = LeeMethod(E_target,resolution,mask,kx,ky)

% for simplicity we take number of DMD pixels equal to the size of E_target
% resolution is given in DMDpixels 

[ny, nx] = size(E_target);

% Create Fourier mask
FourierMaskSystemResolution = FourierMask(ny,nx,resolution,mask);

% Calculate which DMD pixels to turn on according to the Lee method
DMDpixels = phaseAndAmplitude_to_DMDpixels_Lee(E_target,kx,ky);

DMDpattern = abs(DMDpixels);

% First lens
DMDpixels_ft = fftshift(fft2(ifftshift(DMDpixels)));

% Spatial filter
DMDpixels_ft = DMDpixels_ft .* FourierMaskSystemResolution;

% Second lens
E_obtained = fftshift(ifft2(ifftshift(DMDpixels_ft)));

% Intensity efficiency (total intensity in E_obtained / total incident
% intensity)
efficiency = sum(sum(abs(E_obtained / sqrt(nx*ny)).^2));

% Normalize fields
E_obtained = E_obtained/sqrt(sum(sum(abs(E_obtained).^2)));
E_target = E_target/sqrt(sum(sum(abs(E_target).^2)));

% Calculate the fidelity
g = innerProduct(E_target,E_obtained);
fidelity = abs(g)^2;

