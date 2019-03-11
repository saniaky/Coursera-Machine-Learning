function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0;
sigma = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

% The "suggested" values are the only ones that the submit grader will accept.
% values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% errValue = 1000000000;
%
% for i = 1:length(values)
%   fprintf('Done: %f %\n', i/length(values) * 100);
%   C_t = values(i);
%
%   for j = 1:length(values)
%     sigma_t = values(j);
%
%     % Train
%     model = svmTrain(X, y, C_t, @(x1, x2) gaussianKernel(x1, x2, sigma_t));
%
%     % Predict on validation set
%     predictions = svmPredict(model, Xval);
%
%     % Calculate error
%     err = mean(double(predictions ~= yval));
%
%     % Check if new error less than currently with C and sigma
%     if (err < errValue)
%       fprintf('New best values: C = %f, sigma =  %f (new %f / old %f)\n', ...
%         C_t, sigma_t, err, errValue);
%
%       C = C_t;
%       sigma = sigma_t;
%       errValue = err;
%     end
%   end
% end


C = 1.000000, sigma =  0.100000;


% =========================================================================

end
