// Utility functions for linear regression
#include "functions/linear_reg_utils.hpp"

namespace quackml {

void printMatrix(std::vector<std::vector<double>> &matrix) {
    for (auto row : matrix) {
        for (auto element : row) {
            std::cout << element << " ";
        }
        std::cout << "\n";
    }
};

void matrixScalarMultiply(std::vector<std::vector<double>> &matrix, float scalar, std::vector<std::vector<double>> &result) {
    // Multiply each element of a matrix by a scalar
    for (size_t i=0; i<matrix.size(); i++) {
        for (size_t j=0; j<matrix[0].size(); j++) {
            result[i][j] = matrix[i][j] * scalar;
        }
    };
};

void matrixMultiply(std::vector<std::vector<double>> &matrix1, std::vector<std::vector<double>> &matrix2, std::vector<std::vector<double>> &result) {
    // Multiply two matrices
    for (size_t i=0; i<matrix1.size(); i++) {
        for (size_t j=0; j<matrix2[0].size(); j++) {
            for (size_t k=0; k<matrix2.size(); k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
};

void matrixSubtract(std::vector<std::vector<double>> &matrix1, std::vector<std::vector<double>> &matrix2, std::vector<std::vector<double>> &result) {
    for (size_t i=0; i<matrix1.size(); i++) {
        for (size_t j=0; j<matrix1[0].size(); j++) {
            result[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }
};

void matrixAdd(std::vector<std::vector<double>> &matrix1, std::vector<std::vector<double>> &matrix2, std::vector<std::vector<double>> &result) {
    for (size_t i=0; i<matrix1.size(); i++) {
        for (size_t j=0; j<matrix1[0].size(); j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
};

std::vector<std::vector<double>> getGradientND(std::vector<std::vector<double>> &sigma, std::vector<std::vector<double>> &c, std::vector<std::vector<double>> &theta, double lambda) {    
    size_t n = sigma.size();
    size_t d = sigma[0].size();

    // sigma * theta
    std::vector<std::vector<double>> result(d, std::vector<double>(1, 0));
    matrixMultiply(sigma, theta, result);

    // sigma * theta - c
    matrixSubtract(result, c, result);

    // (1/n) * (sigma * theta - c)
    matrixScalarMultiply(result, (1.0 / n), result);

    // lambda * theta
    std::vector<std::vector<double>> regularizer(d, std::vector<double>(1, 0));
    matrixScalarMultiply(theta, lambda, regularizer);

    // (1/n) * (sigma * theta - c) + (lambda * theta)
    matrixAdd(result, regularizer, result);

    return result;
};

double gradientNorm(std::vector<std::vector<double>> &matrix) {
    double sum = 0;
    for (auto row : matrix) {
        sum += row[0] * row[0];
    }
    return std::sqrt(sum);
}

} // namespace quackml