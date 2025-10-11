function conf_interval_b1 = slope_ci(xs, ys, alpha)
    n = length(xs);

    % Create a column of ones and concatenate with xs
    ones_column = ones(size(xs, 1), 1);
    xs_column = xs(:); % Reshape xs to a column vector

    X = [ones_column, xs_column]; % Concatenate

    % Compute the matrix XtX and its inverse
    XtX = X' * X;
    c_matrix = inv(XtX);

    % Initialize H matrix
    H = zeros(n, 1);

    % Compute H values
    for i = 1:n
        H(i) = X(i, :) * (c_matrix * X(i, :)');
    end

    % Compute average of the diagonal elements of H
    h_bar = mean(H);

    % Compute e_ii and d_ii for each i
    e = H / h_bar;
    d = min(4, e);

    % Compute the Ordinary Least Squares (OLS) estimate

    beta_ols = inv(X' * X) * (X' * ys(:));

    y_hat = X * beta_ols;
    residuals = ys(:) - y_hat;

    % Construct diagonal matrix A
    A_diag = residuals.^2 .* (1 - H).^-d;
    A = diag(A_diag);
    
    % Calculate matrix S
    S = c_matrix * (X' * (A * (X * c_matrix)));

    % Extract standard errors for intercept and slope
    S_0 = sqrt(S(1, 1));  % Standard error for intercept
    S_1 = sqrt(S(2, 2));  % Standard error for slope

    % Calculate confidence interval for the slope (b1)
    t_value = tinv(1 - alpha / 2, n - 2);  % t critical value
    b1 = beta_ols(2);
    conf_interval_b1 = [b1 - t_value * S_1, b1 + t_value * S_1];

end

