
function DMDpixels = phaseAndAmplitude_to_DMDpixels_Lee(E,kx,ky)

E = E / max(max(abs(E)));

[ny, nx] = size(E);

referenceBeam = ones(ny,nx);

% Loop over all DMD pixels

for k = 1:ny
    for j = 1:nx
        referenceBeam(k,j) = exp(-2*1i*pi*(k*ky+j*kx+1/2));
        DMDpixels(k,j) = (abs(mod((angle(E(k,j))/(2*pi)+(k*ky+j*kx)),1)-1/2) <= (asin(abs(E(k,j)))/pi/2));
    end
end

% Multiply with phase factor, equivalent to placing aperture off-axis
DMDpixels = DMDpixels.*referenceBeam; 
