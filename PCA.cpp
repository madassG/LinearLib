//
// Created by genuis on 05.04.2022.
//

#include "PCA.h"
#include "operator.cpp"
#include "linear.cpp"

PCA::PCA(Linear::Matrix<double> matrix) : _matrix(std::move(matrix)), _nipled(false)
{
}

void PCA::center() {
    auto [rows, columns] = _matrix.Size();
    for (size_t j = 0; j < columns; ++j) {
        double mean = 0;
        for (size_t i = 0; i < rows; ++i) {
            mean += _matrix.At(i, j);
        }

        mean /= static_cast<double>(rows);

        for (size_t i = 0; i < rows; ++i) {
            _matrix.At(i, j) -= mean;
        }
    }
}

void PCA::scale() {
    auto [rows, columns] = _matrix.Size();
    if (rows < 2) throw std::invalid_argument("Cannot scale matrix with <2 rows!");

    for (size_t j = 0; j < columns; ++j) {
        double mean = 0;
        for (size_t i = 0; i < rows; ++i) {
            mean += _matrix.At(i, j);
        }

        mean /= static_cast<double>(rows);

        double deviation = 0;
        for (size_t i = 0; i < rows; ++i) {
            deviation += std::pow((_matrix.At(i, j) - mean), 2);
            _matrix.At(i, j) -= mean;
        }

        deviation /= static_cast<double>(rows - 1);
        deviation = std::pow(deviation, 0.5);

        for (size_t i = 0; i < rows; ++i) {
            _matrix.At(i, j) /= deviation;
        }
    }
}

std::tuple<Linear::Matrix<double>, Linear::Matrix<double>, Linear::Matrix<double>> PCA::nipals(size_t A, double epsilon) {
    auto [rows, columns] = _matrix.Size();
    Linear::Matrix<double> t(static_cast<int>(rows), 1), p, t_old;
    Linear::Matrix<double> matrix = _matrix;
    std::vector<Linear::Matrix<double>> T, P;

    if (A == -1) {
        A = columns;
    }

    for (size_t k = 0; k < A; ++k) {
            for (size_t i = 0; i < rows; ++i) {
                t.At(i, 0) = matrix.At(i, k);
            }

            do {
                p = t.Transpose().dot(matrix).Transpose();

                double abs = t.Transpose().dot(t).At(0, 0);

                for (size_t i = 0; i < columns; ++i) {
                    p.At(i, 0) /= abs;
                }

                abs = p.Norm();

                for (size_t i = 0; i < columns; ++i) {
                    p.At(i, 0) /= abs;
                }

                t_old = t;

                t = _matrix.dot(p);

                abs = p.Transpose().dot(p).At(0, 0);

                for (size_t i = 0; i < rows; ++i) {
                    t.At(i, 0) /= abs;
                }


        } while ((t_old - t).Norm() > epsilon);

        T.push_back(t);
        P.push_back(p);
        matrix = matrix - t.dot(p.Transpose());
    }

    Linear::Matrix<double> result_T(static_cast<int>(rows), static_cast<int>(T.size())),
                           result_P(static_cast<int>(P[0].Size().first), static_cast<int>(P.size())),
                           result_E(static_cast<int>(rows), static_cast<int>(columns));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < T.size(); ++j) {
            result_T.At(i, j) = T[j].At(i, 0);
            if (i < P[0].Size().first) {
                result_P.At(i, j) = P[j].At(i, 0);
            }

        }
    }

    P_ = result_P;
    T_ = result_T;
    E_ = matrix;
    _nipled = true;

    return {T_, P_, E_};
}

std::vector<double> PCA::scope() const {
    if (!_nipled) throw std::invalid_argument("Apply NIPALS algorithm firstly.");
    auto [rows, columns] = T_.Size();

    std::vector<double> result;

    for (size_t i = 0; i < rows; ++i) {
        Linear::Matrix<double> t(1, static_cast<int>(columns));
        for (size_t j = 0; j < columns; ++j) {
            t.At(0, j) = T_.Sel(i, j);
        }

        auto h = T_.Transpose().dot(T_);
        h = h.Reverse();
        h = t.dot(h);

        h = h.dot(t.Transpose());

        result.push_back(h.At(0, 0));
    }

    return result;
}

std::vector<double> PCA::deviation() const {
    if (!_nipled) throw std::invalid_argument("Apply NIPALS algorithm firstly.");
    auto [rows, columns] = E_.Size();

    std::vector<double> result;

    for (size_t i = 0; i < rows; ++i) {
        double vi = 0;
        for (size_t j = 0; j < columns; ++j) {
            vi += E_.Sel(i, j) * E_.Sel(i, j);
        }

        result.push_back(vi);
    }

    return result;
}

std::vector<double> PCA::dispersion() const {
    if (!_nipled) throw std::invalid_argument("Apply NIPALS algorithm firstly.");
    size_t columns = E_.Size().second;
    std::vector<double> dev = deviation();

    for(double& d : dev) {
        d /= static_cast<double>(columns);
    }

    return dev;
}

double PCA::dispersion_mean() const {
    if (!_nipled) throw std::invalid_argument("Apply NIPALS algorithm firstly.");
    size_t rows = E_.Size().first;
    std::vector<double> dev = deviation();

    double result = 0;

    for (double d : dev) result += d;

    return result / static_cast<double>(rows);
}

double PCA::dispersion_general() const {
    if (!_nipled) throw std::invalid_argument("Apply NIPALS algorithm firstly.");
    size_t rows = E_.Size().first;
    std::vector<double> disp = dispersion();

    double result = 0;

    for (double d : disp) result += d;

    return result / static_cast<double>(rows);
}

double PCA::dispersion_explained() const {
    if (!_nipled) throw std::invalid_argument("Apply NIPALS algorithm firstly.");
    auto [rows, columns] = _matrix.Size();
    size_t e_rows = E_.Size().first;

    double result = 0;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < columns; ++j) {
            result += std::pow(_matrix.Sel(i, j), 2);
        }
    }

    result = 1 - (static_cast<double>(e_rows) * dispersion_mean()) / result;

    return result;

}

void PCA::show() {
    std::cout << _matrix << '\n';
}
