
function [DMDpattern,E_obtained,fidelity,efficiency] = superpixelMethod(E_target,resolution,mask)

% for simplicity we take number of DMD pixels equal to the size of E_target
% resolution is given in DMDpixels 

[ny, nx] = size(E_target);

% Create Fourier mask
FourierMaskSystemResolution = FourierMask(ny,nx,resolution,mask);

% Rescale the target to the superpixel resolution
E_superpixelResolution = rescaleTargetToSuperpixelResolution(E_target);

% Normalize such that the maximum amplitude is 1
E_superpixelResolution = E_superpixelResolution / max(max(abs(E_superpixelResolution)));

% Calculate which DMD pixels to turn on according to the superpixel method
DMDpixels = phaseAndAmplitude_to_DMDpixels_lookupTable(E_superpixelResolution,'maxAmplitude');
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

