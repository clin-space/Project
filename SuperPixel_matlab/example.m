clear all
close all

addpath('misc_functions')
addpath('superpixel_lookupTable')

% Diffraction limited spot size in target plane in units of DMD pixels:
resolution = 8;

% Shape of spatial filter:
mask = 'circle';    % choose 'circle'/'square1'/'square2'/'gaussian'

% Create a target field of the size of the DMD:

[E_target, maxIntensity] = makeFieldFromImage();
%[E_target, maxIntensity] = makeSpecklePattern(8);


% Apply the superpixel method to the target field. Here superpixels of size
% 4x4 are used. 

[DMDpattern_superpixel, E_superpixel, fidelity_superpixel, efficiency_superpixel] = superpixelMethod(E_target, resolution, mask);


% Apply the Lee method to the target field. 
% Spatial carrier frequency in units of DMD pixels:
Lee_k = 1/12;

[DMDpattern_Lee, E_Lee, fidelity_Lee, efficiency_Lee] = LeeMethod(E_target, resolution, mask, Lee_k, Lee_k);


E_target_lowres = spatialFilter(E_target,resolution);
fidelity_superpixel_lowres = abs(innerProduct(E_superpixel,E_target_lowres))^2
fidelity_Lee_lowres = abs(innerProduct(E_Lee,E_target_lowres))^2


% Plot the results:

plot_E(E_target/sqrt(sum(sum(abs(E_target).^2))),maxIntensity,'gray','target');

plot_DMDpixels(DMDpattern_superpixel,'superpixel');
plot_E(E_superpixel,maxIntensity,'gray','superpixel');

plot_DMDpixels(DMDpattern_Lee,'Lee');
plot_E(E_Lee,maxIntensity,'gray','Lee');
