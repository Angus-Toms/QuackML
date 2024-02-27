// linear_reg_utils.hpp 

#ifndef LINEAR_REG_UTILS_HPP
#define LINEAR_REG_UTILS_HPP

#pragma once 
#include <vector>
#include <iostream>
#include <cmath>

namespace quackml {

    // Matrix operations 
    void printMatrix(std::vector<std::vector<double>> &matrix);
    void matrixScalarMultiply(std::vector<std::vector<double>> &matrix, float scalar, std::vector<std::vector<double>> &result);
    void matrixMultiply(std::vector<std::vector<double>> &matrix1, std::vector<std::vector<double>> &matrix2, std::vector<std::vector<double>> &result);
    void matrixSubtract(std::vector<std::vector<double>> &matrix1, std::vector<std::vector<double>> &matrix2, std::vector<std::vector<double>> &result);
    void matrixAdd(std::vector<std::vector<double>> &matrix1, std::vector<std::vector<double>> &matrix2, std::vector<std::vector<double>> &result);

    // Linear regression functions 
    std::vector<std::vector<double>> getGradientND(std::vector<std::vector<double>> &sigma, std::vector<std::vector<double>> &c, std::vector<std::vector<double>> &theta, double lambda);
    double gradientNorm(std::vector<std::vector<double>> &matrix);

} // namespace quackml

#endif // LINEAR_REG_UTILS_HPP