
function gamma = innerProduct(E1,E2)

norm_E1 = sqrt(sum(sum(abs(E1).^2)));
norm_E2 = sqrt(sum(sum(abs(E2).^2)));

gamma = sum(sum( E1 .* conj(E2)));
gamma = gamma / (norm_E1*norm_E2);
