#ifndef CONTROLBASIS_HPP
#define CONTROLBASIS_HPP

#include <vector>
#include <numeric>
#include <assert.h>
#include <iterator>
#include <algorithm>
#include <iostream>

typedef std::vector<double> stdvec;
typedef std::vector< std::vector<double>> rowmat;

class ControlBasis
{
   // control has form u(t_i) = u0(t_i) + S(t_i)*sum(c_1*f_1(t_i) + ... + c_M*f_M(t_i))
private:

    stdvec u0;
    stdvec S;
    stdvec c;
    stdvec ucurrent;
    rowmat f;
    rowmat controlJacobian;
    rowmat vmat;
    size_t M, N;


public:
    ControlBasis();
    ControlBasis(stdvec& u0, stdvec& S, rowmat& f);

    size_t getM() const;
    size_t getN() const;

    stdvec convertControl( const stdvec& control, const bool new_control = true ); 
    stdvec convertGradient( const stdvec& gradu ) const;
    rowmat convertHessian( const rowmat& Hessu ) const;
    rowmat getControlJacobian( ) const;

};

#endif
