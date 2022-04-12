#ifndef LINEAR_EXCEPTION_H
#define LINEAR_EXCEPTION_H

#include <stdexcept>

namespace Linear {
    class MatrixException: public std::exception {
    public:
        explicit MatrixException(const char* message);
        explicit MatrixException(std::string message);
        virtual const char* what() const noexcept;

    protected:
        std::string msg_;
    };

    class IncorrectInitialVector: public MatrixException {
    public:
        explicit IncorrectInitialVector(const char* message);
        explicit IncorrectInitialVector(std::string message);

    };


    class OutOfRange: public MatrixException {
    public:
        explicit OutOfRange(const char* message);
        explicit OutOfRange(std::string message);
    };


    class IncompatibleSize: public MatrixException {
    public:
        explicit IncompatibleSize(const char* message);
        explicit IncompatibleSize(std::string message);
    };


    class WrongFormat: public MatrixException {
    public:
        explicit WrongFormat(const char* message);
        explicit WrongFormat(std::string message);
    };

    class ZeroDeterminant: public MatrixException {
    public:
        explicit ZeroDeterminant(const char* message);
        explicit ZeroDeterminant(std::string message);
    };

}


#endif //LINEAR_EXCEPTION_H
