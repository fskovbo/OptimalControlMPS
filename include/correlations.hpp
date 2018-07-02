#ifndef CORRELATIONS_H
#define CORRELATIONS_H

#include "itensor/all.h"
#include <iostream>

using namespace itensor;


inline Cplx correlationFunction(SiteSet const& sites, IQMPS& psi, std::string const& opname1, int i, std::string const& opname2, int j)
{
  // get operators at site i and j  
  auto op_i = sites.op(opname1,i);
  auto op_j = sites.op(opname2,j);

  if (j == i)
  {
    psi.position(i);

    auto ket = psi.A(i)* op_j*prime(op_i,Site);
    auto bra = prime(prime(dag(psi.A(i)),Site),Site);
    return (bra*ket).real();
  }
  else
  if (j > i) {   }
  else
  if (i > j) {
    int tmp = i;
    i = j;
    j = tmp;
  }

  //'gauge' the MPS to site i
  //any 'position' between i and j, inclusive, would work here
  psi.position(i);

  //psi.Anc(1) *= psi.A(0); //Uncomment if doing iDMRG calculation

  //index linking i to i+1:
  auto ir = commonIndex(psi.A(i),psi.A(i+1),Link);

  auto C = psi.A(i)*op_i*prime(dag(psi.A(i)),Site,ir);
  for(int k = i+1; k < j; ++k)
  {
    C *= psi.A(k);
    C *= prime(dag(psi.A(k)),Link);
  }
  C *= psi.A(j);
  C *= op_j;
  //index linking j to j-1:
  auto jl = commonIndex(psi.A(j),psi.A(j-1),Link);
  C *= prime(dag(psi.A(j)),jl,Site);

  return C.cplx(); //or C.cplx() if expecting complex
}

inline ITensor correlationMatrix(SiteSet const& sites, IQMPS& psi, std::string const& opname1, std::string const& opname2){
  int N = sites.N();
  // initialize indices of correlation matrix
  itensor::Index rho_i("rho i",N);
  itensor::Index rho_j = prime(rho_i);

  ITensor rho(rho_i,rho_j);
  Cplx Cij;
  for (int i = 1; i <= N; ++i)
  {
    // calculate diagonal elements
    Cij = correlationFunction(sites,psi,opname1,i,opname2,i);
    rho.set(rho_i(i),rho_j(i), Cij);

    for (int j = i+1; j <= N; ++j)
    {
      // calculate off-diagonal elements
      Cij = correlationFunction(sites,psi,opname1,i,opname2,j);
      rho.set(rho_i(i),rho_j(j), Cij);
      rho.set(rho_i(j),rho_j(i), conj(Cij));
    }
  }
  return rho;
}

inline Real correlationTerm(SiteSet const& sites, IQMPS& psi, std::string const& opname1, std::string const& opname2)
{
  // get correlation matrix of operators
  auto rho = correlationMatrix(sites,psi,opname1,opname2);

  ITensor V, D;
  // diagonalize correlation matrix
  diagHermitian(rho,V,D, {"Maxm",1});

  auto indices = D.inds();
  auto index1 = indices.index(1);
  auto index2 = indices.index(2);

  // return largest eigenvalue of correlation matrix
  return D.real(index1(1),index2(1));
}

inline Cplx expectationValue(SiteSet const& sites, IQMPS& psi, std::string const& opname, int i)
{
  // get expectation value of operator at site i
  auto op = sites.op(opname,i);
  psi.position(i);
  auto ket = psi.A(i);
  auto bra = dag(prime(ket,Site));
  return (bra*op*ket).cplx();
}

inline std::vector<Cplx> expectationValues(SiteSet const& sites, IQMPS& psi, std::string const& opname)
{
  // get expectation values of operator at each site
  std::vector<Cplx> expVals;
  for (int i = 1; i <= sites.N(); i++) {
    expVals.push_back( expectationValue(sites,psi,opname,i) );
  }
  return expVals;
}

inline std::vector<double> entanglementEntropy(SiteSet const& sites, IQMPS& psi)
{
  auto N = sites.N();
  std::vector<double> Svec;

  for(size_t i = 1; i < N; i++)
  {
    psi.position(i); 

    //Compute two-site wavefunction for sites (i,i+1)
    IQTensor wf = psi.A(i)*psi.A(i+1);

    //SVD this wavefunction to get the spectrum
    //of density-matrix eigenvalues
    auto U = psi.A(i);
    IQTensor S,V;
    auto spectrum = svd(wf,U,S,V);

    //Apply von Neumann formula
    //spectrum.eigs() is a Vector containing
    //the density matrix eigenvalues
    //(squares of the singular values)
    double SvN = 0.;
    for(auto p : spectrum.eigs())
    {
      if(p > 1E-12) SvN += -p*log(p);
    }
    Svec.push_back(SvN);
  }
  return Svec;
}

#endif
