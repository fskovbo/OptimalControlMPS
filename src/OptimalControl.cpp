#include "OptimalControl.hpp"
#include "BH_tDMRG.hpp"

template<class TimeStepper>
OptimalControl<TimeStepper>::
OptimalControl(IQMPS& psi_target, IQMPS& psi_init, TimeStepper& timeStepper, size_t N, double gamma)
  : psi_target(psi_target), psi_init(psi_init), N(N), gamma(gamma),
    timeStepper(timeStepper), tstep(timeStepper.getTstep())
{
  basis = ControlBasis();
  GRAPE = true;
  M     = 0;
}


template<class TimeStepper>
OptimalControl<TimeStepper>::
OptimalControl(IQMPS& psi_target, IQMPS& psi_init, TimeStepper& timeStepper, ControlBasis& basis, double gamma)
  : psi_target(psi_target), psi_init(psi_init), gamma(gamma),
    timeStepper(timeStepper), basis(basis), tstep(timeStepper.getTstep())
{
  GRAPE = false;
  N     = basis.getN();
  M     = basis.getM();  
}


template<class TimeStepper>
double OptimalControl<TimeStepper>::calcRegularization(const stdvec& control) const
{
  double tmp = 0;
  for (size_t i = 0; i < N-1; i++) {

    double diff = control[i+1]-control[i];
    tmp += diff*diff/tstep;
  }

  return gamma/2.0*tmp;
}


template<class TimeStepper>
stdvec OptimalControl<TimeStepper>::calcRegularizationGrad(const stdvec& control) const
{
  std::vector<double> del;
  del.reserve(N);

  del.push_back(-gamma*(-5.0*control[1] + 4.0*control[2] - control[3]
                  + 2.0*control[0])/tstep);

  for (size_t i = 1; i < N-1; i++) {
    del.push_back(-gamma*(control[i+1] + control[i-1] - 2.0*control[i])/tstep);
  }

  del.push_back( -gamma*(-5.0*control[N-2] + 4.0*control[N-3]
                  - control[N-4] + 2.0*control[N-1])/tstep);

  return del;
}


template<class TimeStepper>
std::vector<IQMPS> OptimalControl<TimeStepper>::getPsit() const
{
  return psi_t;
}


template<class TimeStepper>
size_t OptimalControl<TimeStepper>::getM() const
{
  return M;
}


template<class TimeStepper>
size_t OptimalControl<TimeStepper>::getN() const
{
  return N;
}


template<class TimeStepper>
void OptimalControl<TimeStepper>::setGamma( double newgamma )
{
  gamma = newgamma;
}


template<class TimeStepper>
stdvec OptimalControl<TimeStepper>::getControl(const stdvec& control)
{
  if (GRAPE) {
    return control;
  }
  else {
    return basis.convertControl(control);
  }
}


template<class TimeStepper>
stdvec OptimalControl<TimeStepper>::getTimeAxis( ) const
{
  std::vector<double> time;
  double tstep = timeStepper.getTstep();
  double t = 0;
  while( fabs(t - N*tstep) > 1e-2*tstep) 
  {
    time.push_back(t);
    t += tstep;         // could recode to better handle rounding errors
  }

  return time;
}


template<class TimeStepper>
stdvec OptimalControl<TimeStepper>::calcFidelityGrad(const stdvec& control, const bool new_control)
{
  if (new_control){
    calcPsi(control);
  }

  auto chi = Cplx_i*psi_target;
  auto overlapFactor = overlapC(psi_t.back(),psi_target);

  std::vector<double> g;
  g.reserve(N);
  g.push_back( -(overlapC( chi , timeStepper.propagatorDeriv(control.back()) , psi_t.back() )*overlapFactor ).real() );

  for (size_t i = N-1; i > 0; i--) {
    timeStepper.step(chi,control[i],control[i-1],false);
    g.push_back( -(overlapC( chi , timeStepper.propagatorDeriv(control[i-1]) , psi_t[i-1] )*overlapFactor ).real() );
  }

  std::reverse(g.begin(),g.end());

  return g;
}

template<class TimeStepper>
rowmat OptimalControl<TimeStepper>::calcHessian(const stdvec& control, const bool new_control){

    rowmat Hessian(N, std::vector<double>(N, 0));

    calcPsi(control);
    calcXi(control);
    auto overlapFactor = overlapC(psi_t.back(),psi_target);

    for(size_t i=0;i<N;++i){
        auto psiH = exactApplyMPO(timeStepper.propagatorDeriv(control[i]),psi_t[i],timeStepper.getArgs()); // Consider the norm, maybe a problem?
        auto normiH = psiH.norm();
        // Calculate diagonal term
        auto xiH = exactApplyMPO(timeStepper.propagatorDeriv(control[i]),xi_t[i],timeStepper.getArgs()); // Could be optimized away
        double val1 = -(overlapFactor*overlapC(xiH,psiH)).imag();
        double val2 = -(overlapC(xi_t[i],timeStepper.propagatorDeriv(control[i]),psi_t[i])*overlapC(psi_t[i],timeStepper.propagatorDeriv(control[i]),xi_t[i])).real();
        Hessian[i][i] = tstep*tstep*(val1+val2);
        // Off diagonal terms
        for(size_t j = i+1;j<N;++j){
            timeStepper.step(psiH,control[j-1],control[j],true);
            auto xiH = exactApplyMPO(timeStepper.propagatorDeriv(control[j]),xi_t[j],timeStepper.getArgs()); // Could be optimized away
            double val1 = -(overlapFactor*overlapC(xiH,psiH)*normiH).imag();
            double val2 = -(overlapC(xi_t[j],timeStepper.propagatorDeriv(control[j]),psi_t[j])*overlapC(psi_t[i],timeStepper.propagatorDeriv(control[i]),xi_t[i])).real();
            Hessian[i][j] = tstep*tstep*(val1+val2);
            Hessian[j][i] = Hessian[i][j];
        }
    }
    return Hessian;
}

template<class TimeStepper>
void OptimalControl<TimeStepper>::calcPsi(const stdvec& control)
{
  const bool propagateForward = true;
  auto psi0 = psi_init;
  psi_t.clear();
  psi_t.push_back(psi0);

  for (size_t i = 0; i < N-1; i++) {
    timeStepper.step(psi0,control[i],control[i+1],propagateForward);
    psi_t.push_back(psi0);
  }
}

template<class TimeStepper>
void OptimalControl<TimeStepper>::calcXi(const stdvec& control)
{
  const bool propagateForward = false;
  auto xiT = psi_target;
  xi_t.clear();
  xi_t.push_back(xiT);

  for (size_t i = N-1; i > 0; i--) {
    timeStepper.step(xiT,control[i],control[i-1],propagateForward);
    xi_t.push_back(xiT);
  }

  std::reverse(xi_t.begin(),xi_t.end());
}

template<class TimeStepper>
double OptimalControl<TimeStepper>::calcCost(const stdvec& control, const bool new_control)
{
  if (new_control) {
    calcPsi(control);
  }

  double re, im;
  overlap(psi_target,psi_t.back(),re,im);

  return 0.5*(1.0-(re*re+im*im)) + calcRegularization(control);
}


template<class TimeStepper>
stdvec OptimalControl<TimeStepper>::calcAnalyticGradient(const stdvec& control, const bool new_control)
{
  auto FGrad = calcFidelityGrad(control,new_control);
  auto RGrad = calcRegularizationGrad(control);

  for (size_t i = 0; i < N; i++) {
    FGrad[i] += RGrad[i];
  }

  return FGrad;
}


template<class TimeStepper>
stdvec OptimalControl<TimeStepper>::calcFidelityForAllT(const stdvec& control, const bool new_control)
{
  if (new_control) {
    calcPsi(control);
  }
  
  std::vector<double> fid;
  fid.reserve(N);

  double re, im;
  for (size_t i = 0; i < N; i++) {
    overlap(psi_target,psi_t[i],re,im);
    fid.push_back(re*re+im*im);
  }

  return fid;
}


template<class TimeStepper>
void OptimalControl<TimeStepper>::propagatePsi(const stdvec& control)
{
  if (GRAPE) {
    calcPsi(control);
  }
  else {
    calcPsi(basis.convertControl(control));
  }
  
}


template<class TimeStepper>
double OptimalControl<TimeStepper>::getCost(const stdvec& control, const bool new_control)
{
  if (GRAPE) {
    return calcCost(control,new_control);
  }
  else {
    return calcCost(basis.convertControl(control,new_control),new_control);
  }
}


template<class TimeStepper>
stdvec OptimalControl<TimeStepper>::getAnalyticGradient(const stdvec& control, const bool new_control)
{
  if (GRAPE) {
    return calcAnalyticGradient(control,new_control);
  }
  else {
    return basis.convertGradient(
                      calcAnalyticGradient(basis.convertControl(control,new_control),new_control)
                                );
  }
}

template<class TimeStepper>
rowmat OptimalControl<TimeStepper>::getHessian(const stdvec& control, const bool new_control)
{
  if (GRAPE) {
    return calcHessian(control,new_control);
  }
//  else {
//    return basis.convertGradient(
//                      calcAnalyticGradient(basis.convertControl(control,new_control),new_control)
//                                );
//  }
}

template<class TimeStepper>
stdvec OptimalControl<TimeStepper>::getFidelityForAllT(const stdvec& control, const bool new_control)
{
  if (GRAPE) {
    return calcFidelityForAllT(control,new_control);
  }
  else {
    return calcFidelityForAllT(basis.convertControl(control,new_control),new_control);
  }
}


template<class TimeStepper>
rowmat OptimalControl<TimeStepper>::getControlJacobian( ) const
{
  if (GRAPE) {
    rowmat cJac(N, std::vector<double>(N, 0));
    
    for(size_t i = 0; i < N; i++)
    {
      cJac[i][i] = 1;
    }
    
    return cJac;
  }
  else {
    return basis.getControlJacobian();
  }
}


template class OptimalControl<BH_tDMRG>;
