#ifndef OPTIMALCONTROL_HPP
#define OPTIMALCONTROL_HPP

#include "itensor/all.h"
#include "ControlBasis.hpp"

#include <vector>
#include <iterator>
#include <algorithm>



using namespace itensor;
using stdvec = std::vector<double>;

template<class TimeStepper>
class OptimalControl{
private:

  TimeStepper& timeStepper;
  double gamma, tstep;
  IQMPS psi_target, psi_init;

  std::vector<IQMPS> psi_t;

  double getRegularization(const stdvec& control);
  stdvec getRegularizationGrad(const stdvec& control);
  stdvec getFidelityGrad(const stdvec& control, const bool new_control);

public:
  OptimalControl(IQMPS& psi_target, IQMPS& psi_init, TimeStepper& timeStepper, double gamma);

  std::vector<IQMPS> getPsit() const;

  // Optimal control using time-descretized control 
  void calcPsi(const stdvec& control);
  double getCost(const stdvec& control, const bool new_control = true);
  stdvec getAnalyticGradient(const stdvec& control, const bool new_control = true);
  stdvec getNumericGradient(const stdvec& control);
  stdvec getFidelityForAllT(const stdvec& control, const bool new_control = true);

  // Optimal control using user-defined control parameterization
  void calcPsi(const ControlBasis& basis);
  double getCost(const ControlBasis& basis, const bool new_control = true);
  stdvec getAnalyticGradient(const ControlBasis& basis, const bool new_control = true);
  stdvec getNumericGradient(const ControlBasis& BasicSiteSet);
  stdvec getFidelityForAllT(const ControlBasis& basis, const bool new_control = true);

  
};

#endif
