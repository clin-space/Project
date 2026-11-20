
function [E_target,maxIntensity] = makeSpecklePattern(resolution)

nx = 1024;
ny = 768;  

FourierMaskSystemResolution = randn(ny,nx)+1i*randn(ny,nx);
maskCenterX = ceil((nx+1)/2);
maskCenterY = ceil((ny+1)/2);

for i = 1:ny
    for j = 1:nx
        if (i-maskCenterY)^2+(ny/nx*(j-maskCenterX))^2 >= (ny/resolution/2)^2
            FourierMaskSystemResolution(i,j) = 0;
        end
    end
end

% Fourier transform back to DMD plane:
E_target = fftshift(ifft2(ifftshift( FourierMaskSystemResolution )));

I = abs(E_target).^2;
maxIntensity = max(max(I))/sum(sum(I));