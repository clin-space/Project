
function E_systemResolution = spatialFilter(E_target, resolution)

[ny,nx] = size(E_target);

FourierMaskSystemResolution = zeros(ny,nx);
maskCenterX = ceil((nx+1)/2);
maskCenterY = ceil((ny+1)/2);

for i = 1:ny
    for j = 1:nx
        if (i-maskCenterY)^2+(ny/nx*(j-maskCenterX))^2 < (ny/resolution/2)^2
                FourierMaskSystemResolution(i,j) = 1;
        end
    end
end
    

E_target_ft = fftshift(fft2(ifftshift(E_target)));
% Apply mask:
E_target_ft = FourierMaskSystemResolution.*E_target_ft;
% Fourier transform back to DMD plane:
E_systemResolution = fftshift(ifft2(ifftshift( E_target_ft )));