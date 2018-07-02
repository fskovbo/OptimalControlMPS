#ifndef INITIALIZESTATE_HPP
#define INITIALIZESTATE_HPP

#include "itensor/all.h"

// Methods for setting up ground states through the ITensor DMRG,
//  allowing cleaner code in mainfiles.
// The methods distribute the particles initially, sets up the Hamiltonian
//  of the desired state, then executes the DMRG algorithm and return the result.
//
// NOTE: for large/complex systems the parameters of the DMRG algorithm may need adjustment 


namespace itensor{


// Finds ground state of Bose-Hubbard Hamiltonian at given J, U
inline IQMPS InitializeState(const SiteSet& sites, const int Npart, const double J, const double U, bool silent = true)
{
  if (silent) std::cout.setstate(std::ios_base::failbit); // silences the DMRG info
  
  int N = sites.N();
  auto state = InitState(sites);
  int p = Npart;

  // initial guess for distribution of particles
  if (Npart > N) { printf("Npart > N not supported\n"); }
  for(int i = N; i >= 1; --i)
      {
      if (p >= 1) {
        state.set(i,"Occ1");
        p -= 1;
      }
      else
          {
          state.set(i,"Emp");
          }
      }
  auto psi = IQMPS(state);

  // construct BH Hamiltonian
  auto ampo = AutoMPO(sites);
  for(int i = 1; i < N; ++i) {
    ampo += -J,"A",i,"Adag",i+1;
    ampo += -J,"Adag",i,"A",i+1;
  }
  for(int i = 1; i <= N; ++i) {
    ampo += 0.5*U,"N(N-1)",i;
  }
  auto H = IQMPO(ampo);

  // set DMRG sweep setting
  auto sweeps = Sweeps(10);
  sweeps.maxm() = 10,20,50,100,200;
  sweeps.cutoff() = 1E-9;
  sweeps.niter() = 2;
  sweeps.noise() = 1E-7,1E-8,0.0;

  // compute ground state of Hamiltonian
  auto energy = dmrg(psi,H,sweeps,{"Quiet",true});

  std::cout.clear(); // clears silence        
  
  return psi;
}


// Finds ground state of Bose-Hubbard Hamiltonian at given J, U
inline IQMPS InitializeState(const SiteSet& sites, const int Npart, const double J,const double U,
                              const int maxBondDim, const double threshold, bool silent = true)
{
  if (silent) std::cout.setstate(std::ios_base::failbit); // silences the DMRG info
  
  int N = sites.N();
  auto state = InitState(sites);
  int p = Npart;

  // initial guess for distribution of particles
  if (Npart > N) { printf("Npart > N not supported\n"); }
  for(int i = N; i >= 1; --i)
      {
      if (p >= 1) {
        state.set(i,"Occ1");
        p -= 1;
      }
      else
          {
          state.set(i,"Emp");
          }
      }
  auto psi = IQMPS(state);

  // construct BH Hamiltonian
  auto ampo = AutoMPO(sites);
  for(int i = 1; i < N; ++i) {
    ampo += -J,"A",i,"Adag",i+1;
    ampo += -J,"Adag",i,"A",i+1;
  }
  for(int i = 1; i <= N; ++i) {
    ampo += 0.5*U,"N(N-1)",i;
  }
  auto H = IQMPO(ampo);

  // set DMRG sweep setting
  auto sweeps = Sweeps(10);
  sweeps.maxm() = 10,20,50,maxBondDim;
  sweeps.cutoff() = threshold;
  sweeps.niter() = 2;
  sweeps.noise() = 1E-7,1E-8,0.0;

  // compute ground state of Hamiltonian
  auto energy = dmrg(psi,H,sweeps,{"Quiet",true});

  std::cout.clear(); // clears silence        
  
  return psi;
}


} // end namespace
#endif
