function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% Part 2: Implement the backpropagation algorithm to compute the gradients

a1 = [ones(m, 1) X];
a2 = [ones(m, 1) sigmoid(a1 * Theta1')];
sig = sigmoid(a2 * Theta2');
yk = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
J = sum(sum(-yk .* log(sig) - (1 - yk) .* log(1 - sig)))/m;
J = J + lambda/(2*m) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));

% -------------------------------------------------------------

d1 = zeros(size(Theta1));
d2 = zeros(size(Theta2));

for t = 1:m,
	dlt3 = sig(t,:)' - yk(t,:)';
	dlt2 = Theta2' * dlt3 .* sigmoidGradient([1; Theta1 * a1(t,:)']);

    d2 = d2 + dlt3 * a2(t,:);
	d1 = d1 + dlt2(2:end) * a1(t,:);
end;

Theta1_grad = (1/m)*d1;
Theta2_grad = (1/m)*d2;

Theta1_grad(1:end,2:end) += (lambda/m)*Theta1(:, 2:end);
Theta2_grad(1:end,2:end) += (lambda/m)*Theta2(:, 2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
