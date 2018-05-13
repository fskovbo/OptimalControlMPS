#ifndef BH_TDMRG_HPP
#define BH_TDMRG_HPP

#include "itensor/all.h"

using namespace itensor;
using GateList = std::vector< BondGate<IQTensor> >;

class BH_tDMRG {
private:
  double tstep, J;

  GateList JGates_tforwards;
  GateList JGates_tbackwards;

  std::vector<IQTensor> UGates1;
  std::vector<IQTensor> UGates2;

  SiteSet sites;
  Args args;

  IQMPO propDeriv;

  void initJGates(const double J);
  void initUGates(const double Ufrom, const double Uto);
  void doStep(IQMPS& psi, const GateList& JGates);

public:
  BH_tDMRG(const SiteSet& sites, const double J, const double tstep, const Args& args);
  void setTstep(const double tstep_);
  void step(IQMPS& psi, const double from, const double to, bool propagateForward = true);
  IQMPO propagatorDeriv(const double& control_n);
  double getTstep();
};

#endif
