function [J, grad] = lrCostFunction(theta, X, y, lambda)


% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


J = (-y'*log(sigmoid(X*theta))-(1-y)'*log(1-sigmoid(X*theta)))/m+sum(theta(2:length(theta)).^2)*lambda/(2*m);

grad = X'*(sigmoid(X*theta)-y)/m;
grad(2:length(theta)) += lambda*theta(2:length(theta))/m;

% =============================================================

grad = grad(:);

end
