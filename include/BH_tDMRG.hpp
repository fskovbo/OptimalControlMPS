#ifndef BH_TDMRG_HPP
#define BH_TDMRG_HPP

#include "itensor/all.h"

using namespace itensor;
using GateList  = std::vector< BondGate<IQTensor> >;
using UGatePair = std::pair<std::vector<IQTensor>,std::vector<IQTensor> >;

// tDMRG propagator for Bose-Hubbard model
// Utilizes a Suzuki-Trotter expansion of propagator:
//    prop = exp(-i H_U(to) dt/2) exp(-i H_J dt) exp(-i H_U(from) dt/2)
// where the interaction U is the control parameter, which is evaluated
// through a split-step: U(t_i) -> U(t_i+1)  

class BH_tDMRG {
private:
  double tstep, J;

  GateList JGates_tforwards;
  GateList JGates_tbackwards;

  SiteSet sites;
  Args args;

  IQMPO propDeriv;

  void initJGates(const double J);
  void initUGates(UGatePair& UGates, const double Ufrom, const double Uto) const;
  void doStep(IQMPS& psi, const UGatePair& UGates, const GateList& JGates) const;

public:
  BH_tDMRG(const SiteSet& sites, const double J, const double tstep, const Args& args);
  void setTstep(const double tstep_);
  void step(IQMPS& psi, const double from, const double to, bool propagateForward = true) const;
  IQMPO propagatorDeriv(const double& control_n) const;
  double getTstep() const;
  Args getArgs() const;
};

#endif
