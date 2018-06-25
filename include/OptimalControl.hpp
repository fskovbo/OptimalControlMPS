#ifndef OPTIMALCONTROL_HPP
#define OPTIMALCONTROL_HPP

#include "itensor/all.h"
#include "ControlBasis.hpp"

#include <vector>
#include <iterator>
#include <algorithm>


using namespace itensor;
using stdvec = std::vector<double>;
using rowmat = std::vector< std::vector<double>>;


template<class TimeStepper>
class OptimalControl{
private:
  TimeStepper timeStepper;
  ControlBasis basis;
  double gamma, tstep;
  size_t N, M, threadCount;
  IQMPS psi_target, psi_init;

  std::vector<IQMPS> psi_t;
  std::vector<IQMPS> xi_t;
  std::vector<Cplx> divT;

  bool GRAPE, BFGS, calculatedXi;
  // calculatedXi tells whether Xi+divT have been calculated for same control


  void    calcPsi(const stdvec& control);
  void    calcXi(const stdvec& control);
  void    calcDivT(const stdvec& control);
  void    calcPsiXiDivT(const stdvec& control);
  double  calcCost(const stdvec& control, const bool new_control = true);
  double  calcRegularization(const stdvec& control) const;
  stdvec  calcRegularizationGrad(const stdvec& control) const;
  rowmat  calcRegularizationHessian(const stdvec& control) const;
  stdvec  calcFidelityGrad(const stdvec& control, const bool new__tcontrol = true);
  stdvec  calcAnalyticGradient(const stdvec& control, const bool new_control = true);
  rowmat  calcHessian(const stdvec& control, const bool new_control = true);
  stdvec  calcFidelityForAllT(const stdvec& control, const bool new_control = true);
  void calcHessianRow(size_t rowIndex, const stdvec& control, const std::vector<IQMPS>& xiHlist, Cplx overlapFactor, rowmat& Hessian);

public:
  OptimalControl(IQMPS& psi_target, IQMPS& psi_init, TimeStepper& timeStepper, size_t N, double gamma, bool BFGS = false);
  OptimalControl(IQMPS& psi_target, IQMPS& psi_init, TimeStepper& timeStepper, ControlBasis& basis, double gamma, bool BFGS = false);

  std::vector<IQMPS> getPsit() const;
  size_t getM() const;
  size_t getN() const;
  stdvec getControl( const stdvec& control );
  stdvec getTimeAxis( ) const;
  void setGamma( double newgamma );
  void setThreadCount(const size_t newThreadCount);
  void setGRAPE(const bool useGRAPE);
  bool useBFGS() const;
 
  void propagatePsi(const stdvec& control);
  double getCost(const stdvec& control, const bool new_control = true);
  stdvec getAnalyticGradient(const stdvec& control, const bool new_control = true);
  rowmat getHessian(const stdvec& control, const bool new_control = true);
  stdvec getFidelityForAllT(const stdvec& control, const bool new_control = true);
  rowmat getControlJacobian() const;
  
};

#endif
