function FourierMask = FourierMask(ny,nx,resolution,mask)

% Create circular aperture around the center

FourierMask = zeros(ny,nx);

maskCenterX = ceil((nx+1)/2);
maskCenterY = ceil((ny+1)/2);

if strcmp(mask,'gaussian')
    for i = 1:ny
        for j = 1:nx    
                FourierMask(i,j) = exp(-((i-maskCenterY)^2+(ny/nx*(j-maskCenterX))^2)/(ny/resolution/2)^2);
        end
    end
elseif strcmp(mask,'square1')
    for i = 1:ny
        for j = 1:nx
            if max(abs(i-maskCenterY),abs(ny/nx*(j-maskCenterX))) < (ny/resolution/2)
                FourierMask(i,j) = 1;
            end
        end
    end
elseif strcmp(mask,'square2')
    for i = 1:ny
        for j = 1:nx
            if abs(i-maskCenterY)+abs(ny/nx*(j-maskCenterX)) < (ny/resolution/2)
                FourierMask(i,j) = 1;
            end
        end
    end
else
    for i = 1:ny
        for j = 1:nx
            if (i-maskCenterY)^2+(ny/nx*(j-maskCenterX))^2 < (ny/resolution/2)^2
                FourierMask(i,j) = 1;
            end
        end
    end
    
end
