#include <utility>

#include "exception.h"


Linear::MatrixException::MatrixException(const char* message) : msg_(message) {}
Linear::MatrixException::MatrixException(std::string message) : msg_(std::move(message)) {}
const char* Linear::MatrixException::what() const noexcept {
    return msg_.c_str();
}

Linear::IncorrectInitialVector::IncorrectInitialVector(const char *message) : Linear::MatrixException(message) {}
Linear::IncorrectInitialVector::IncorrectInitialVector(std::string message):
        Linear::MatrixException(std::move(message)) {}

Linear::OutOfRange::OutOfRange(const char *message) : Linear::MatrixException(message) {}
Linear::OutOfRange::OutOfRange(std::string message) : Linear::MatrixException(std::move(message)) {}


Linear::IncompatibleSize::IncompatibleSize(const char *message) : Linear::MatrixException(message) {}
Linear::IncompatibleSize::IncompatibleSize(std::string message) :
        Linear::MatrixException(std::move(message)) {}

Linear::WrongFormat::WrongFormat(const char *message) : Linear::MatrixException(message) {}
Linear::WrongFormat::WrongFormat(std::string message) : Linear::MatrixException(std::move(message)) {}

Linear::ZeroDeterminant::ZeroDeterminant(const char *message) : Linear::MatrixException(message) {}
Linear::ZeroDeterminant::ZeroDeterminant(std::string message) : Linear::MatrixException(std::move(message)) {}
