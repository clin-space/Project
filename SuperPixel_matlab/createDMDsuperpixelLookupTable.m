
% superpixel size:
n = 4;

m = n^2;

basepoints = exp(1i*(0:(m-1))/m*2*pi);
%basepoints = exp(33/32*1i*(0:(m-1))/m*2*pi);

% figure(1)
% hold on
% for i=1:m
%     plot(real(basepoints(i)),imag(basepoints(i)),'o','MarkerSize',4,'MarkerEdgeColor','k','MarkerFaceColor','k');
% end
% 
% axis([-1.5,1.5,-1.5,1.5])
% xlabel('Re(E)','FontSize',16);
% ylabel('Im(E)','FontSize',16);

figure(2)
hold on

targetFields = zeros(2^m,m+1);

numbers = (1:2^m)';

for i=1:m
    targetFields(:,1) = targetFields(:,1) + mod(numbers,2)*basepoints(i);
    targetFields(:,i+1) = mod(numbers,2);
    numbers = floor(numbers/2);
end

targetFields(:,1) = round(targetFields(:,1)*1e10)/1e10;

tic;
targetFields = sortrows(targetFields,1);
toc

i = 2;
N_targetFields = length(targetFields');

while i <= N_targetFields
    if targetFields(i,1) == targetFields(i-1,1)
        if sum(targetFields(i,2:m+1)) > sum(targetFields(i-1,2:m+1))
            targetFields = targetFields([1:i-1 i+1:N_targetFields],:);
            N_targetFields = N_targetFields-1;
        else
            targetFields = targetFields([1:i-2 i:N_targetFields],:);
            N_targetFields = N_targetFields-1;
        end
    else
        i=i+1
        N_targetFields
    end
end

targetFields(:,1) = targetFields(:,1)/n;
plot(targetFields(:,1),'o','MarkerSize',3,'MarkerEdgeColor','k','MarkerFaceColor','k')

xlabel('Re(E)','FontSize',26);
ylabel('Im(E)','FontSize',26);

maxAmplitude = max(abs(targetFields(:,1)));
stepSize = 0.003;

linearGrid = [-fliplr(stepSize:stepSize:maxAmplitude) 0:stepSize:maxAmplitude];
N = length(linearGrid);
amplitudeGrid = ones(N,1)*linearGrid + 1i*linearGrid'*ones(1,N);
lookupTable = zeros(N,N);

tic;

for re = 1:N
    for im = 1:N
        [y,index] = min(abs(targetFields(:,1)-amplitudeGrid(im,re)));
        lookupTable(im,re) = index;
        ((re-1)*N+im)/N^2
    end
end
toc

dlmwrite('targetFields',targetFields);
dlmwrite('lookupTable',lookupTable);
dlmwrite('gridParameters',[maxAmplitude stepSize]);