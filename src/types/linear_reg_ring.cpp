#include "types/linear_reg_ring.hpp"

#include <iostream>
#include <stdexcept>    

// MUNGO TODO: Remove some of these constructors
// ASSUMPTIONS:
// We assume that each input relation contains labels
// All relation feature-lists are disjoint 
// Relations are joined on a non-feature non-label column (e.g. id)

namespace quackml {

// Constructor from feature observations, used by to_ring
LinRegRing::LinRegRing(std::vector<duckdb::vector<duckdb::Value>> &observations) {
    auto d = observations[0].size();
    count = observations.size();
    sums = std::vector<double>(d, 0);
    covar = std::vector<std::vector<double>>(d);

    // Sums
    for (idx_t i = 0; i < d; i++) {
        double sum = 0;
        for (auto &obs : observations) {
            sum += obs[i].GetValue<double>();
        }
        sums[i] = sum;
    }

    // Covariance 
    for (idx_t i = 0; i < d; i++) {
        auto row = std::vector<double>(d-i, 0);
        for (idx_t j = i; j < d; j++) {
            double sum = 0;
            for (auto &obs : observations) {
                sum += obs[i].GetValue<double>() * obs[j].GetValue<double>();
            }
            row[j-i] = sum;
        }
        covar[i] = row;
    } 
}

// Direct constructor used by linear_reg_ring
LinRegRing::LinRegRing(double count, std::vector<double> sums, std::vector<std::vector<double>> covar) {
    this->count = count;
    this->sums = sums;
    this->covar = covar;
}

// Copy constructor
LinRegRing::LinRegRing(const LinRegRing &other) {
    count = other.count;
    sums = other.sums;
    covar = other.covar;
}

// Operators
LinRegRing &LinRegRing::operator+(const LinRegRing &other) {
    if (sums.size() != other.sums.size() || covar.size() != other.covar.size()) {
        throw std::invalid_argument("LinRegRing operator+ sizes do not match");
    }
    // Count 
    count += other.count;

    // Sums
    for (idx_t i = 0; i < sums.size(); i++) {
        sums[i] += other.sums[i];
    }

    // Covar
    for (idx_t i = 0; i < covar.size(); i++) {
        for (idx_t j = 0; j < covar[i].size(); j++) {
            covar[i][j] += other.covar[i][j];
        }
    }
    return *this;
}

LinRegRing &LinRegRing::operator*(const LinRegRing &other) {
    if (sums.size() != other.sums.size() || covar.size() != other.covar.size()) {
        throw std::invalid_argument("LinRegRing operator* sizes do not match");
    }

    // covar[i][j] = (other_count * covar[i][j]) + (count * other.covar[i][j]) 
    //              + (sums[i] * other.sums[j]) + (sums[j] * other.sums[i])
    // Note: covar is stored as upper triangle of matrix
    for (idx_t i = 0; i < covar.size(); i++) {
        for (idx_t j = 0; j < covar[i].size(); j++) {
            covar[i][j] = (other.count * covar[i][j])
                        + (count * other.covar[i][j]) 
                        + (sums[i+j] * other.sums[i])
                        + (sums[i] * other.sums[i+j]);
        }
    }

    // sums = (other_count * sums) + (count * other_sums)
    for (idx_t i = 0; i < sums.size(); i++) {
        sums[i] = (other.count * sums[i]) + (count * other.sums[i]);
    }

    // count = count * other_count
    count *= other.count;

    return *this;
}


void LinRegRing::padLower(idx_t inc) {
    // Add 0s to sums 
    for (idx_t i = 0; i < inc; i++) {
        sums.push_back(0);
    }

    // Add 0s to end of every covar row 
    for (auto &row : covar) {
        for (idx_t i = 0; i < inc; i++) {
            row.push_back(0);
        }
    }

    // Add rows of 0s to covar 
    // Note: covar is stored as upper triangle of matrix so rows decrease in size 
    for (idx_t i = inc; i > 0; i--) {
        auto row = std::vector<double>(i, 0);
        covar.push_back(row);
    }
}

void LinRegRing::padUpper(idx_t inc) {
    auto d = sums.size();
    // Add 0s to top of sums 
    for (idx_t i = 0; i < inc; i++) {
        sums.insert(sums.begin(), 0);
    }

    // Add rows of 0s to covar 
    // Note: covar is stored as upper triangle of matrix so rows decrease in size 
    for (idx_t i = 0; i < inc; i++) {
        auto row = std::vector<double>(d+i+1, 0);
        covar.insert(covar.begin(), row);
    }
}

void LinRegRing::print() {
    std::cout << "Count: " << count << "\n";
    std::cout << "Sums: ";
    for (auto &sum : sums) {
        std::cout << sum << " ";
    }
    std::cout << "\n";
    std::cout << "Covar:\n";
    for (auto &row : covar) {
        for (auto &val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}


} // namespace quackml
