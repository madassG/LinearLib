#ifndef LINEAR_PCA_H
#define LINEAR_PCA_H

#include <utility>

#include "linear.h"
#include <string>
#include <iostream>


class PCA {
public:
    explicit PCA(Linear::Matrix<double> matrix);

    void center();
    void scale();
    std::tuple<Linear::Matrix<double>, Linear::Matrix<double>, Linear::Matrix<double>> nipals(size_t A = -1, double epsilon = 0.00001);
    [[nodiscard]] std::vector<double> scope() const;
    [[nodiscard]] std::vector<double> deviation() const;
    [[nodiscard]] std::vector<double> dispersion() const;
    [[nodiscard]] double dispersion_mean() const;
    [[nodiscard]] double dispersion_general() const;
    [[nodiscard]] double dispersion_explained() const;

    void show();
private:
    Linear::Matrix<double> _matrix, P_, T_, E_;
    bool _nipled;
};


#endif
