function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% size(X)
% size(centroids)
% X - 300x2
% cnetroids - 3x2

m = size(X);
for i = 1:m
  p1 = X(i, :);
  c = 1; % default cluster index
  c_distance = sum((p1 - centroids(1, :)) .^ 2);
  for j = 2:K
    p2 = centroids(j, :);
    tmp = sum((p1 - p2) .^ 2);
    %fprintf('i = %d, dist = %f -> %f, cluster - %d\n', i, c_distance, tmp, j);
    if tmp <= c_distance
      c = j;
      c_distance = tmp;
    end
  end
  idx(i) = c;
end



% =============================================================

end
