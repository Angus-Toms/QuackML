// Ridge linear regression with L2 cost function, optimised with GD using covariance matrix.
// Then: map this functionality onto the initialise, combine, finalise etc. methods of duckdb UDAFs

#include <vector>
#include <iostream>

float getGradient1D(size_t n, int Sigma, int C, float theta, float lambda) {
    // Calculate cost function gradient using covariance matrix
    return (1.0 / n) * ((Sigma * theta - C) + (lambda * theta));
};

float linearRegression1D(std::vector<int> features, std::vector<int> labels, float alpha, float lambda, size_t iter) {
    int n = features.size();
    // Weight initialization
    float theta = 1.0;
    // Get covar matrix
    int Sigma = 0;
    int C = 0;
    for (size_t i=0; i<n; i++) {
        Sigma += features[i] * features[i];
        C += features[i] * labels[i];
    }

    // Gradient descent
    for (size_t i=0; i<iter; i++) {
        // Calculate gradient of cost function 
        float gradient = getGradient1D(n, Sigma, C, theta, lambda);
        // Update theta 
        theta = theta - alpha * gradient;
    }

    // Predictions 
    std::cout << "Predictions: ";
    for (auto feature : features) {
        std::cout << feature * theta << ", ";
    }

    return theta;
};

void matrixScalarMultiply(std::vector<std::vector<float>> &matrix, float scalar, std::vector<std::vector<float>> &result) {
    // Multiply each element of a matrix by a scalar
    for (size_t i=0; i<matrix.size(); i++) {
        for (size_t j=0; j<matrix[0].size(); j++) {
            result[i][j] = matrix[i][j] * scalar;
        }
    };
};

void matrixMultiply(std::vector<std::vector<float>> &matrix1, std::vector<std::vector<float>> &matrix2, std::vector<std::vector<float>> &result) {
    // Multiply two matrices
    for (size_t i=0; i<matrix1.size(); i++) {
        for (size_t j=0; j<matrix2[0].size(); j++) {
            for (size_t k=0; k<matrix2.size(); k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
};

void matrixSubtract(std::vector<std::vector<float>> &matrix1, std::vector<std::vector<float>> &matrix2, std::vector<std::vector<float>> &result) {
    for (size_t i=0; i<matrix1.size(); i++) {
        for (size_t j=0; j<matrix1[0].size(); j++) {
            result[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }
};

void matrixAdd(std::vector<std::vector<float>> &matrix1, std::vector<std::vector<float>> &matrix2, std::vector<std::vector<float>> &result) {
    for (size_t i=0; i<matrix1.size(); i++) {
        for (size_t j=0; j<matrix1[0].size(); j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
};

void printMatrix(std::vector<std::vector<float>> &matrix) {
    for (auto row : matrix) {
        for (auto element : row) {
            std::cout << element << ", ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
};

std::vector<std::vector<float>> getGradientND(std::vector<std::vector<float>> &sigma, std::vector<std::vector<float>> &c, std::vector<std::vector<float>> &theta, float lambda) {
    size_t n = sigma.size();
    size_t d = sigma[0].size();
    
    // Sigma * theta
    std::vector<std::vector<float>> result(d, std::vector<float>(1, 0));
    matrixMultiply(sigma, theta, result);

    // Sigma * theta - C
    matrixSubtract(result, c, result);

    // (1/n) * (Sigma * theta - C)
    matrixScalarMultiply(result, (1.0 / n), result);

    // lambda * theta
    std::vector<std::vector<float>> regularizer(d, std::vector<float>(1, 0));
    matrixScalarMultiply(theta, lambda, regularizer);

    // (1/n) * (Sigma * theta - C) + (lambda * theta)
    matrixAdd(result, regularizer, result);

    return result;
};

void linearRegressionND(std::vector<std::vector<float>> &features, std::vector<std::vector<float>> &labels, float alpha, float lambda, size_t iter) {
    size_t n = features.size();
    size_t d = features[0].size();

    std::vector<std::vector<float>> theta(d, std::vector<float>(1, 1.0)); // d, - column vector
    std::vector<std::vector<float>> sigma(d, std::vector<float>(d, 0)); // d,d - matrix
    std::vector<std::vector<float>> c(d, std::vector<float>(1, 0)); // d, - column vector
    for (size_t i=0; i<d; i++) {
        for (size_t k=0; k<n; k++) {
            c[i][0] += features[k][i] * labels[k][0];
            for (size_t j=0; j<d; j++) {
                sigma[i][j] += features[k][i] * features[k][j];
            }
        }
    }

    for (size_t i=0; i<iter; i++) {
        auto gradient = getGradientND(sigma, c, theta, lambda);
        matrixScalarMultiply(gradient, alpha, gradient);
        matrixSubtract(theta, gradient, theta);
    }

    std::cout << "Predictions: ";
    for (auto feature : features) {
        float prediction = 0;
        for (size_t i=0; i<d; i++) {
            prediction += feature[i] * theta[i][0];
        }
        std::cout << prediction << ", ";
    }

    std::cout << "\nWeights: ";
    for (size_t i=0; i<d; i++) {
        std::cout << theta[i][0] << ", ";
    }
}

int main() {
    std::vector<std::vector<float>> features = {
        {2, 5},
        {3, 1},
        {-1, 4}
    };
    float theta1 = -2.486;
    float theta2 = 0.477;
    std::vector<std::vector<float>> labels = {
        {theta1 * features[0][0] + theta2 * features[0][1]},
        {theta1 * features[1][0] + theta2 * features[1][1]},
        {theta1 * features[2][0] + theta2 * features[2][1]}
    };

    float alpha = 0.01;
    float lambda = 0.0;
    linearRegressionND(features, labels, alpha, lambda, 1000);
};