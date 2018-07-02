#ifndef CONTROLBASISFACTORY_HPP
#define CONTROLBASISFACTORY_HPP

#include "ControlBasis.hpp"
#include "SeedGenerator.hpp"
#include <vector>
#include <assert.h>
#include "math.h"

#define PI 3.14159265
typedef std::vector<double> stdvec;
typedef std::vector< std::vector<double>> rowmat;


class ControlBasisFactory{

private:

public:

  static ControlBasis buildChoppedSineBasis(stdvec& u0, double tstep, double T, size_t M);

};

ControlBasis ControlBasisFactory::buildChoppedSineBasis(stdvec& u0, double tstep, double T, size_t M)
{
  size_t N = u0.size();
  assert( N-(1 + T/tstep) < 1e-5 );

  auto x    = SeedGenerator::linspace(0,100,N);

  // build Shape functions
  auto S    = SeedGenerator::sigmoid(x,8.0,1.1);
  auto S2   = SeedGenerator::sigmoid(x,-8.0,100-1.1);

  for (size_t i = N/2; i < N; i++) {
    S.at(i) = S2.at(i);
  }
  S.at(0)   = 0;
  S.at(N-1) = 0;

  // build sin( omega_n*pi*t/T ) matrix
  rowmat f(N, std::vector<double>(M, 0));      
  for(size_t i = 0; i < N; i++)
  {
    for(size_t n = 0; n < M; n++)
    {
      f[i][n] = sin( (n+1)*PI*tstep*i/T);
    }   
  }

  return ControlBasis(u0,S,f);
}


#endif