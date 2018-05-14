#include "OptimalControl.hpp"
#include "BH_tDMRG.hpp"

template<class TimeStepper>
OptimalControl<TimeStepper>::OptimalControl(IQMPS& psi_target, IQMPS& psi_init, TimeStepper& timeStepper, double gamma)
  : psi_target(psi_target), psi_init(psi_init), gamma(gamma), timeStepper(timeStepper), tstep(timeStepper.getTstep())
{
}


template<class TimeStepper>
double OptimalControl<TimeStepper>::getRegularization(const stdvec& control)
{
  double tmp = 0;
  for (size_t i = 0; i < control.size()-1; i++) {

    double diff = control.at(i+1)-control.at(i);
    tmp += diff*diff/tstep;
  }

  return gamma/2.0*tmp;
}


template<class TimeStepper>
stdvec OptimalControl<TimeStepper>::getRegularizationGrad(const stdvec& control)
{
  std::vector<double> del;
  del.reserve(control.size());

  del.push_back(-gamma*(-5.0*control.at(1) + 4.0*control.at(2) - control.at(3)
                  + 2.0*control.at(0))/tstep);

  for (size_t i = 1; i < control.size()-1; i++) {
    del.push_back(-gamma*(control.at(i+1) + control.at(i-1) - 2.0*control.at(i))/tstep);
  }

  del.push_back( -gamma*(-5.0*control.at(control.size()-2) + 4.0*control.at(control.size()-3)
                  - control.at(control.size()-4) + 2.0*control.at(control.size()-1))/tstep);

  return del;
}


template<class TimeStepper>
std::vector<IQMPS> OptimalControl<TimeStepper>::getPsit() const
{
  return psi_t;
}


template<class TimeStepper>
stdvec OptimalControl<TimeStepper>::getFidelityGrad(const stdvec& control, const bool new_control)
{
  if (new_control){
    calcPsi(control);
  }

  auto chi = Cplx_i*psi_target;
  auto overlapFactor = overlapC(psi_t.back(),psi_target);

  std::vector<double> g;
  g.reserve(control.size());
  g.push_back( -(overlapC( chi , timeStepper.propagatorDeriv(control.back()) , psi_t.back() )*overlapFactor ).real() );

  for (size_t i = control.size()-1; i > 0; i--) {
    timeStepper.step(chi,control.at(i),control.at(i-1),false);
    g.push_back( -(overlapC( chi , timeStepper.propagatorDeriv(control.at(i-1)) , psi_t.at(i-1) )*overlapFactor ).real() );
  }

  std::reverse(g.begin(),g.end());

  return g;
}


template<class TimeStepper>
void OptimalControl<TimeStepper>::calcPsi(const stdvec& control)
{
  const bool propagateForward = true;
  auto psi0 = psi_init;
  psi_t.clear();
  psi_t.push_back(psi0);

  for (size_t i = 0; i < control.size()-1; i++) {
    timeStepper.step(psi0,control.at(i),control.at(i+1),propagateForward);
    psi_t.push_back(psi0);
  }
}


template<class TimeStepper>
double OptimalControl<TimeStepper>::getCost(const stdvec& control, const bool new_control)
{
  if (new_control) {
    calcPsi(control);
  }

  double re, im;
  overlap(psi_target,psi_t.back(),re,im);

  return 0.5*(1.0-(re*re+im*im)) + getRegularization(control);
}


template<class TimeStepper>
stdvec OptimalControl<TimeStepper>::getAnalyticGradient(const stdvec& control, const bool new_control)
{
  auto FGrad = getFidelityGrad(control,new_control);
  auto RGrad = getRegularizationGrad(control);

  for (size_t i = 0; i < FGrad.size(); i++) {
    FGrad.at(i) += RGrad.at(i);
  }

  return FGrad;
}


template<class TimeStepper>
stdvec OptimalControl<TimeStepper>::getNumericGradient(const stdvec& control)
{
  auto ccopy = control;
  double Jp, Jm, epsilon = 1e-5;
  std::vector<double> g;
  g.reserve(control.size());

  size_t count = 0;

  for (auto& ui : ccopy){
    ui        += epsilon;
    Jp         = getCost(ccopy);

    ui        -= 2.0*epsilon;
    Jm         = getCost(ccopy);

    ui        += epsilon;

    std::cout << "Calculated derivative nr. " << count++ << '\n';

    g.push_back((Jp-Jm)/(2.0*epsilon));
  }
  
  return g;
}


template<class TimeStepper>
stdvec OptimalControl<TimeStepper>::getFidelityForAllT(const stdvec& control, const bool new_control)
{
  if (new_control) {
    calcPsi(control);
  }
  
  std::vector<double> fid;
  fid.reserve(control.size());

  double re, im;
  for (size_t i = 0; i < control.size(); i++) {
    overlap(psi_target,psi_t.at(i),re,im);
    fid.push_back(re*re+im*im);
  }

  return fid;
}


template<class TimeStepper>
void OptimalControl<TimeStepper>::calcPsi(const ControlBasis& basis)
{
  calcPsi(basis.convertControl());
}


template<class TimeStepper>
double OptimalControl<TimeStepper>::getCost(const ControlBasis& basis, const bool new_control)
{
  return getCost(basis.convertControl(),new_control);
}


template<class TimeStepper>
stdvec OptimalControl<TimeStepper>::getAnalyticGradient(const ControlBasis& basis, const bool new_control)
{
  return basis.convertBackGradient( getAnalyticGradient(basis.convertControl(),new_control) );
}


template<class TimeStepper>
stdvec OptimalControl<TimeStepper>::getNumericGradient(const ControlBasis& basis)
{
  auto basiscopy  = basis;
  auto cArray     = basiscopy.getCArray();
  double Jp, Jm, epsilon = 1e-5;
  std::vector<double> g;

  for (auto& ci : cArray){
    ci        += epsilon;
    basiscopy.setCArray(cArray);
    Jp         = getCost(basiscopy);

    ci        -= 2.0*epsilon;
    basiscopy.setCArray(cArray);
    Jm         = getCost(basiscopy);

    ci        += epsilon;
    basiscopy.setCArray(cArray);
    g.push_back((Jp-Jm)/(2.0*epsilon));
  }
  
  return g;
}


template<class TimeStepper>
stdvec OptimalControl<TimeStepper>::getFidelityForAllT(const ControlBasis& basis, const bool new_control)
{
  return getFidelityForAllT(basis.convertControl(),new_control);
}


template class OptimalControl<BH_tDMRG>;
