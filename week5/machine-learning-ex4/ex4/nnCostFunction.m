function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% Perform feedforward propagation:
a1 = [ones(m, 1) X];
a2 = sigmoid(a1 * Theta1'); a2 = [ones(m, 1) a2];
a3 = sigmoid(a2 * Theta2');
h = a3;

% m - 5000
% num_labels - 10
% h - 5000 x 10
% y - 5000 x 1  ->  5000 x 10


% Convert vector y as vectors containing 0 or 1
% so it has ones in the corresponding column,
% e.g. [ 0 0 0 0 0 0 0 0 0 1 ]
% for label of digit zero and the vertical size
% of y is m, i.e. number of examples.
yt = zeros(m, num_labels);
for i = 1:m
  yt(i, y(i)) = 1;
end
%yt(1999:2002, :) transition from 3 to 4


% Cost Function, non-regularized:
total = 0;
for i = 1:m
  for k = 1:num_labels
    total = total - yt(i, k) .* log(h(i, k)) - (1 - yt(i, k)) .* log(1 - h(i, k));
  end
end
J = 1/m * total;
%fprintf('--> Unregularized cost: %f\n', J);
% cost is about 0.287629


% Regularize cost function
% Theta1 - 25 x 401
total1 = 0;
for i = 1:size(Theta1, 1)
  for k = 2:size(Theta1, 2)
    total1 = total1 + Theta1(i, k) ^ 2;
  end
end
% Theta2 - 10 x 26
total2 = 0;
for i = 1:size(Theta2, 1)
  for k = 2:size(Theta2, 2)
    total2 = total2 + Theta2(i, k) ^ 2;
  end
end
J = J + lambda / (2 * m) * (total1 + total2);
%fprintf('--> Regularized cost: %f\n', J);
% Expected cost is about 0.383770


% Backpropagation algorithm
%fprintf('--> Running backpropagation algorigthm\n');
fprintf('yt: %fx%f\n', size(yt,1), size(yt,2)); %
yt(1, :)
Delta2 = zeros(size(Theta2));
Delta1 = zeros(size(Theta1));
for i = 1:m
  % Perform "forward" pass
  a1 = [1; X(i, :)'];

  z2 = Theta1 * a1;
  a2 = [1; sigmoid(z2)];

  z3 = Theta2 * a2;
  a3 = sigmoid(z3);

  % fprintf('a1: %fx%f\n', size(a1,1), size(a1,2)); % a1: 5000x401
  % fprintf('a2: %fx%f\n', size(a2,1), size(a2,2)); % a2: 5000x26
  % fprintf('a3: %fx%f\n', size(a3,1), size(a3,2)); % a3: 5000x10
  % fprintf('z2: %fx%f\n', size(z2,1), size(z2,2)); % z2: 5000x25
  % fprintf('Theta2: %fx%f\n', size(Theta2,1), size(Theta2,2)); % Theta2, Delta2 and Theta2grad: 10x26
  % fprintf('Theta1: %fx%f\n', size(Theta1,1), size(Theta1,2)); % Theta1, Delta1 and Theta1grad: 25x401

  % Compute error terms (gradients)
  yVector = (1:num_labels)' == y(i);
  d3 = a3 - yVector;
  % fprintf('d3: %fx%f\n', size(d3,1), size(d3,2)); % d3: 5000x10

  % d2 equals the product of δ3 and Θ2 (ignoring the Θ2 bias units),
  % then multiplied element-wise by the g′() of z2 (computed back in Step 2).
  d2 = Theta2' * d3;
  d2 = d2(2:end) .* sigmoidGradient(z2); % hidden_layer_size x 1 == 25 x 1 %Removing delta2 for bias node
  % fprintf('d2: %fx%f\n', size(d2,1), size(d2,2)); % d2: 5000x25

  % Accumulate sum
  Delta2 = Delta2 + d3 * a2'; % 10 x 26
  Delta1 = Delta1 + d2 * a1'; % 25 x 401

  % fprintf('Delta2: %fx%f\n', size(Delta2,1), size(Delta2,2)); % 10 x 26
  % fprintf('Delta1: %fx%f\n', size(Delta1,1), size(Delta1,2)); % 25 x 401
end


Delta2(:,2:end) = Delta2(:,2:end) + lambda * Theta2(:,2:end);
Delta1(:,2:end) = Delta1(:,2:end) + lambda * Theta1(:,2:end);

Theta2_grad = Delta2 / m;
Theta1_grad = Delta1 / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
