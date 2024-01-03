// Ridge linear regression with L2 cost function, optimised with GD using covariance matrix.
// Then: map this functionality onto the initialise, combine, finalise etc. methods of duckdb UDAFs

#include <vector>
#include <iostream>

float get_gradient(std::vector<int> features, std::vector<int> labels, float theta, float lambda) {
    // Calculate cost function gradient using covariance matrix
    int feature_count = features.size();

    int Sigma = 0;
    int C = 0;
    for (int i = 0; i < feature_count; i++) {
        Sigma += features[i] * features[i];
        C += features[i] * labels[i];
    }

    return (1.0 / feature_count) * ((Sigma * theta - C) + (lambda * theta));
};

float linear_regression_1d(std::vector<int> features, std::vector<int> labels, float alpha, float lambda, int gd_iterations) {
    // Weight initialization
    float theta = 1.0;

    // Gradient descent
    for (int i=0; i<gd_iterations; i++) {
        // Calculate gradient of cost function 
        float gradient = get_gradient(features, labels, theta, lambda);

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

int main() {
    std::vector<int> test_features {1, 2, 3, 4, 5};
    std::vector<int> test_labels {3, 6, 9, 12, 15};
    float theta = linear_regression_1d(test_features, test_labels, 0.1, 1, 100);
    std::cout << "Theta: " << theta << std::endl;
};