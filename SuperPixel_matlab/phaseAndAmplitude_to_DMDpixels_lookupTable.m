function DMDpixels = phaseAndAmplitude_to_DMDpixels_lookupTable(E,normalization)

m = 4;

% Load lookup table
targetFields = dlmread('targetFields');
lookupTable = dlmread('lookupTable');
gridParameters = dlmread('gridParameters');
maxAmplitude = gridParameters(1);
stepSize = gridParameters(2);
lookupTable_x0 = (length(lookupTable)+1)/2;

[ny,nx] = size(E);

DMDpixels = zeros(ny*m,nx*m);

% Decrease maximum amplitude to 1 if needed
E = exp(angle(E)*1i).*min(abs(E),1);

% Correct for overall phase offset
E = E.*exp(11/16*pi*1i);

% Choose normalization: maxAmplitude for highest efficiency, highRes to
% restrict the modulation to a smaller and denser disk in the complex plane
switch normalization
    case 'maxAmplitude'
        E = E * maxAmplitude;
    case 'highRes'
        E = E * 0.906131;       % 4 pixel, to be calibrated for different superpixel sizes
end

% Loop over superpixels. Find correct combination of pixels to turn on in the lookup table and put
% them into the 'DMDpixels' matrix that contains the DMD pattern
E = E / stepSize;



for j = 0:ny-1
    for i = 0:nx-1
        idx = lookupTable(round(imag(E(j+1,i+1))+lookupTable_x0),round(real(E(j+1,i+1))+lookupTable_x0));
        pixels = targetFields(idx,2:m^2+1);
        shift = mod(m*j,m^2);
        pixels = [pixels(shift+1:m^2) pixels(1:shift)];
        DMDpixels(m*j+1:m*j+m,m*i+1:m*i+m) = reshape(pixels,m,m);
    end
end

% Apply phase gradient to DMD pixels (equivalent to placing Fourier mask
% off-axis):
phaseFactor = ones(ny*m,nx*m);
for k = 1:ny*m
    for j = 1:nx*m
        phaseFactor(k,j) = exp(1i*pi*(k+4*j)/8);
    end
end
DMDpixels = DMDpixels.*phaseFactor;

