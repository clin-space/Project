function E_superpixelResolution = rescaleTargetToSuperpixelResolution(E_target)

% Rescale the target field to the superpixel resolution (currently
% only 4x4 superpixels implemented)
superpixelSize = 4;

[ny,nx] = size(E_target);
maskCenterX = ceil((nx+1)/2);
maskCenterY = ceil((ny+1)/2);

FourierMaskSuperpixelResolution = zeros(ny,nx);

nSuperpixelX = nx / superpixelSize;
nSuperpixelY = ny / superpixelSize;

for i = 1:ny
    for j = 1:nx
        if (i-maskCenterY)^2+(ny/nx*(j-maskCenterX))^2 < (ny/superpixelSize/2)^2
            FourierMaskSuperpixelResolution(i,j) = 1;
        end
    end
end

E_target_ft = fftshift(fft2(ifftshift(E_target)));

% Apply mask
E_target_ft = FourierMaskSuperpixelResolution.*E_target_ft;

% Remove zeros outside of mask
E_superpixelResolution_ft = E_target_ft((maskCenterY - ceil((nSuperpixelY-1)/2)):(maskCenterY + floor((nSuperpixelY-1)/2)),(maskCenterX - ceil((nSuperpixelX-1)/2)):(maskCenterX + floor((nSuperpixelX-1)/2)));

% Add phase gradient to compensate for anomalous 1.5 pixel shift in real
% plane
phaseFactor = ones(nSuperpixelY,nSuperpixelX);
for k = 1:nSuperpixelY
    for j = 1:nSuperpixelX
        phaseFactor(k,j) = exp(2*1i*pi*(k/nSuperpixelY+j/nSuperpixelX)*3/8);
    end
end
E_superpixelResolution_ft = E_superpixelResolution_ft.*phaseFactor;

% Fourier transform back to DMD plane
E_superpixelResolution = fftshift(ifft2(ifftshift( E_superpixelResolution_ft )));
