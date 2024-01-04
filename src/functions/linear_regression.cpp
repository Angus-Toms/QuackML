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

float* matrixScalarMultiply(const float *pMatrix, size_t rows, size_t cols, float scalar, float *pResult) {
    // Multiply each element of a matrix by a scalar
    for (size_t i=0; i<rows; i++) {
        for (size_t j=0; j<cols; j++) {
            pResult[i * cols + j] = pMatrix[i * cols + j] * scalar;
        }
    }
    return pResult;
};

float* matrixMultiply(const float *pMatrix1, size_t rows1, size_t cols1, const float *pMatrix2, size_t rows2, size_t cols2, float *pResult) {
    // Matrix multiplication
    assert(cols1 == rows2);

    for (size_t i=0; i<rows1; i++) {
        for (size_t j=0; j<cols2; j++) {
            pResult[i * cols2 + j] = 0;
            for (size_t k=0; k<cols1; k++) {
                pResult[i * cols2 + j] += pMatrix1[i * cols1 + k] * pMatrix2[k * cols2 + j];
            }
        }
    }
    return pResult;
}

float* matrixSubtract(const float *pMatrix1, const float *pMatrix2, size_t rows, size_t cols, float *pResult) {
    // Subtract matrix 2 from matrix 1
    for (size_t i=0; i<rows; i++) {
        for (size_t j=0; j<cols; j++) {
            pResult[i * cols + j] = pMatrix1[i * cols + j] - pMatrix2[i * cols + j];
        }
    }
    return pResult;
};

float* matrixAdd(const float *pMatrix1, const float *pMatrix2, size_t rows, size_t cols, float *pResult) {
    // Add matrix 2 to matrix 1
    for (size_t i=0; i<rows; i++) {
        for (size_t j=0; j<cols; j++) {
            pResult[i * cols + j] = pMatrix1[i * cols + j] + pMatrix2[i * cols + j];
        }
    }
    return pResult;
};

float* getGradientND(size_t n, size_t d, float **pSigma, float *pC, float *pTheta, float lambda) {
    // n: number of observations, d: dimensions (variables) in each feature
    // Calculate cost function gradient using covariance matrix
    
    // Sigma * theta
    float *result = new float[d];
    matrixMultiply(*pSigma, d, d, pTheta, d, 1, result);
    // Sigma * theta - C
    matrixSubtract(result, pC, d, 1, result);
    // (1/n) * (Sigma * theta - C)
    matrixScalarMultiply(result, d, 1, (1.0 / n), result);
    // lambda * theta
    float *pRegularizer = new float[d];
    matrixScalarMultiply(pTheta, 1, d, lambda, pRegularizer);
    // (1/n) * (Sigma * theta - C) + (lambda * theta)
    matrixAdd(result, pRegularizer, d, 1, result);
    return result;
};

float* linearRegressionND(float **pFeatures, float *pLabels, float alpha, float lambda, size_t iter, size_t n, size_t d) {
    // Weight initialization
    float *pTheta = new float[d];
    std::fill(pTheta, pTheta + d, 1.0);

    // Get covar matrix
    float **pSigma = new float*[d];
    for (size_t i=0; i<d; i++) {
        pSigma[i] = new float[d];
        std::fill(pSigma[i], pSigma[i] + d, 0.0);
    }

    float *pC = new float[d];
    std::fill(pC, pC + d, 0.0);
    for (size_t i=0; i<d; i++) {
        for (size_t j=i; j<d; j++) {
            for (size_t k=0; k<n; k++) { // Iterate over n observations
                pSigma[i][j] += pFeatures[k][i] * pFeatures[k][j];
                pC[i] += pFeatures[k][i] * pLabels[k];
            }
        }
    }

    // Gradient descent 
    for (size_t i=0; i<iter; i++) {
        auto gradient = getGradientND(n, d, pSigma, pC, pTheta, lambda);
        matrixScalarMultiply(gradient, d, 1, alpha, gradient);
        matrixSubtract(pTheta, gradient, d, 1, pTheta);
    }

    // Predictions
    std::cout << "Predictions: ";
    for (size_t i=0; i<n; i++) {
        float prediction = 0.0;
        for (size_t j=0; j<d; j++) {
            prediction += pFeatures[i][j] * pTheta[j];
        }
        std::cout << prediction << ", ";
    }
    return pTheta;
}

int main() {
    size_t n = 3;
    size_t d = 2;

    float **pFeatures = new float*[n];
    for (size_t i=0; i<n; i++) {
        pFeatures[i] = new float[d];
    }
    pFeatures[0][0] = 1.0;
    pFeatures[0][1] = 3.0;
    pFeatures[1][0] = 9.0;
    pFeatures[1][1] = 4.0;
    pFeatures[2][0] = -5.0;
    pFeatures[2][1] = 2.0;

    float *pLabels = new float[3];
    pLabels[0] = 9.0;
    pLabels[1] = 35.0;   
    pLabels[2] = 11.0;

    auto theta = linearRegressionND(pFeatures, pLabels, 0.1, 0, 10, n, d);
    
    std::cout << "\nTheta: ";
    for (size_t i=0; i<d; i++) {
        std::cout << theta[i] << ", ";
    }

    // Deallocate memory
    for (size_t i=0; i<n; i++) {
        delete[] pFeatures[i];
    }
    delete[] pFeatures;
};