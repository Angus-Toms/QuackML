#include "duckdb.hpp"

#include <vector>
#include <iostream>
#include <stdexcept>    

// ASSUMPTIONS:
// We assume that each input relation contains labels
// All relation feature-lists are disjoint 
// Relations are joined on a non-feature non-label column (e.g. id)

namespace quackml {

static void printMatrix(std::vector<std::vector<double>> &myMatrix) {
    // Debugging tool
    std::cout << "-------------\n";
    for (auto row : myMatrix) {
        for (auto element : row) {
            std::cout << element << ", ";
        }
        std::cout << "\n";
    }
    std::cout << "-------------\n";
};

struct LinearRegressionRingElement : duckdb::LogicalType {
private:
    double count;
    std::vector<double> sums;
    std::vector<std::vector<double>> covar;

public:
    idx_t d;
    LinearRegressionRingElement() {
        count = 0;
    }
    // Create ring with dxd covariance matrix of 0s
    LinearRegressionRingElement(idx_t d) {
        this->d = d;
        count = 0;
        sums = std::vector<double>(d, 0);
        covar = std::vector<std::vector<double>>(d, std::vector<double>(d, 0));
    }
    LinearRegressionRingElement(std::vector<double> features) {
        count = 1;
        features.insert(features.begin(), 0);
        sums = features;
        
        // Instantiate covariance matrix
        d = features.size();
        for (idx_t i = 0; i < d; i++) {
            auto row = new std::vector<double>(d, 0);
            (*row)[i] = features[i] * features[i]; 
            covar.push_back(*row);
        }
        printMatrix(covar);
    }
    ~LinearRegressionRingElement() {}

    LinearRegressionRingElement operator+(const LinearRegressionRingElement &other) {
        if (d != other.d) {
            throw std::invalid_argument("Cannot add LinearRegressionRingElements with different dimensions");
        }

        LinearRegressionRingElement result;

        // Counts
        result.count = count + other.count;

        // Sums and covariance matrix
        for (idx_t i = 0; i < d; i++) {
            result.sums[i] = sums[i] + other.sums[i];
            for (idx_t j = 0; j < d; j++) {
                result.covar[i][j] = covar[i][j] + other.covar[i][j];
            }
        }
        return result;
    }

    // HMMMMMM: return to this.
    LinearRegressionRingElement operator*(const LinearRegressionRingElement &other) {
        // Create covar matrix of combined features
        LinearRegressionRingElement result(d + other.d);

        // Counts
        result.count = count * other.count;

        // Sums 
        for (idx_t i = 0; i < d; i++) {
            result.sums[i] = other.sums[i] * count + sums[i] * other.count;
        }

        // Covariance matrix 
        // See: https://www.youtube.com/watch?v=W6GrRmqvLyc&list=PL7dGmiBS0RtfVhhedAG7OzrL6nygjgGyn&index=3
        for (idx_t i = 0; i < d; i++) {
            for (idx_t j = 0; j < d; j++) {
                result.covar[i][j] = (other.count * covar[i][j]) + (count * other.covar[i][j]) + (sums[i] * other.sums[j]) + (other.sums[i] * sums[j]);
            }
        }
        return result;
    }

    double getCount() {
        return count;
    }

    std::vector<double>* getSums() {
        return &sums;
    }

    std::vector<std::vector<double>>* getCovar() {
        return &covar;
    }
};

} // namespace quackml