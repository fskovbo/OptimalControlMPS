#include "OptimalControl.hpp"
#include "BH_tDMRG.hpp"

#include <thread>
#include <mutex>

template<class TimeStepper>
OptimalControl<TimeStepper>::
OptimalControl(IQMPS& psi_target, IQMPS& psi_init, TimeStepper& timeStepper, size_t N, double gamma, bool BFGS)
  : psi_target(psi_target), psi_init(psi_init), N(N), gamma(gamma),
    timeStepper(timeStepper), tstep(timeStepper.getTstep()), BFGS(BFGS)
{
  basis         = ControlBasis();
  GRAPE         = true;
  calculatedXi  = false;
  threadCount   = 2;
  M             = 0;
}


template<class TimeStepper>
OptimalControl<TimeStepper>::
OptimalControl(IQMPS& psi_target, IQMPS& psi_init, TimeStepper& timeStepper, ControlBasis& basis, double gamma, bool BFGS)
  : psi_target(psi_target), psi_init(psi_init), gamma(gamma),
    timeStepper(timeStepper), basis(basis), tstep(timeStepper.getTstep()), BFGS(BFGS)
{
  GRAPE         = false;
  calculatedXi  = false;
  N             = basis.getN();
  M             = basis.getM();
  threadCount   = 2;
}

template<class TimeStepper>
void OptimalControl<TimeStepper>::setThreadCount(const size_t newThreadCount){
    if(newThreadCount < 2) throw std::invalid_argument("Mininum threadCount is 2.");
    threadCount = newThreadCount;
}
template<class TimeStepper>
void OptimalControl<TimeStepper>::setGRAPE(const bool useGRAPE){
    GRAPE = useGRAPE;
    calculatedXi = false;
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
rowmat OptimalControl<TimeStepper>::calcRegularizationHessian(const stdvec& control) const
{
  rowmat Hessian(N, std::vector<double>(N, 0));
  
  for(size_t i = 1; i < N-1; i++)
  {
    Hessian[i][i-1] = -gamma/tstep;
    Hessian[i][i+1] = -gamma/tstep;
    Hessian[i][i]   = 2.0*gamma/tstep;
  }
  Hessian[1][0]=0;
  Hessian[N-2][N-1]=0;

  //Hessian[0][0]     = 2.0*gamma/tstep;
  //Hessian[0][1]     = -gamma/tstep;
  //Hessian[N-1][N-1] = 2.0*gamma/tstep;
  //Hessian[N-1][N-2] = -gamma/tstep;
  
  return Hessian;
}

template<class TimeStepper>
bool OptimalControl<TimeStepper>::useBFGS() const
{
    return BFGS;
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
  if (new_control)
  {
    calculatedXi = false;
    if (BFGS) calcPsi(control);
    else      calcPsiXiDivT(control);
  }
  
  if (BFGS)
  {        
      auto xi = psi_target;
      auto Toverlap = overlapC( xi , timeStepper.propagatorDeriv(control.back()) , psi_t.back() ) ;
      divT.push_back(Toverlap);
      
      for(size_t i = N-1; i > 0; i--)
      {
        timeStepper.step(xi,control[i],control[i-1],false);      
        auto Toverlap = overlapC( xi , timeStepper.propagatorDeriv(control[i-1]) , psi_t[i-1] ) ;
        divT.push_back(Toverlap);
      }
      std::reverse(divT.begin(),divT.end());
  }
  else
  {
    if(!calculatedXi) // True implies BFGS is false
    {
      calcXi(control);
      calcDivT(control);
    }
  }

  // Calculate the gradient
  std::vector<double> g;
  g.reserve(N);
  auto overlapFactor = overlapC(psi_t.back(),psi_target);
  for(size_t i = 0; i < N; ++i)
  {
    g.push_back(tstep * (divT[i]*overlapFactor*Cplx_i).real() );
  }

  return g;
}

template<class TimeStepper>
void OptimalControl<TimeStepper>::calcHessianRow(size_t rowIndex, const stdvec& control, const std::vector<IQMPS>& xiHlist, Cplx overlapFactor, rowmat& Hessian)
{
  auto psiH = exactApplyMPO(timeStepper.propagatorDeriv(control[rowIndex]),psi_t[rowIndex],timeStepper.getArgs());
  auto normiH = norm(psiH);

  // Calculate diagonal term
  double val1 = (overlapFactor*overlapC(xiHlist[rowIndex],psiH)).real();
  double val2 = -( divT[rowIndex] * conj(divT[rowIndex]) ).real();
  Hessian[rowIndex][rowIndex] += tstep*tstep*(val1+val2);

  // Off diagonal terms
  for(size_t j = rowIndex+1; j < N - 1; ++j)
  {
    timeStepper.step(psiH,control[j-1],control[j],true);
    double val1 = (overlapFactor*overlapC(xiHlist[j],psiH)*normiH).real();
    double val2 = -( divT[rowIndex] * conj(divT[j]) ).real();
    Hessian[rowIndex][j] += tstep*tstep*(val1+val2);
    Hessian[j][rowIndex] += tstep*tstep*(val1+val2); // dont calculate edges
  }
}

template<class TimeStepper>
rowmat OptimalControl<TimeStepper>::calcHessian(const stdvec& control, const bool new_control)
{
  if (new_control)
  {
    calculatedXi = false;
    calcPsiXiDivT(control);
  }
  if (!calculatedXi)
  {
    calcXi(control);
    calcDivT(control);
  }

  rowmat Hessian      = calcRegularizationHessian(control);
  auto overlapFactor  = overlapC(psi_t.back(),psi_target);
  std::vector<IQMPS> xiHlist;
  xiHlist.reserve(N);
  
  for(size_t i = 0; i < N; i++)
  {
    xiHlist.push_back( exactApplyMPO(timeStepper.propagatorDeriv(control[i]),xi_t[i],timeStepper.getArgs()) );
  }

  //size_t threadCount = 8;
  size_t nextHessianValueIndex = 1; // Do not calculate the first row, which is on the Hessian Boundary.
  std::vector<std::thread> hessianThreads;
  std::mutex syncMutex;
  hessianThreads.reserve(threadCount); 

  auto threadMainCode = [&syncMutex, &nextHessianValueIndex, &control, &xiHlist, overlapFactor, &Hessian, this]()
  {
    while(nextHessianValueIndex < N - 1)
    {
      size_t currentIndex;
      syncMutex.lock();
      currentIndex = nextHessianValueIndex;
      nextHessianValueIndex++;
      syncMutex.unlock();
      if (currentIndex < N - 1) // Check that another thread did not just took the last job.
      {
        calcHessianRow(currentIndex, control, xiHlist, overlapFactor, Hessian);
      }
    }
  };

  for (size_t i = 0; i < threadCount; ++i)
  {
    hessianThreads.push_back(std::thread(threadMainCode));
  }
  for (size_t i = 0; i < threadCount; ++i)
  {
    hessianThreads[i].join();
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
  calculatedXi = false;
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
  calculatedXi = true;
}

template<class TimeStepper>
void OptimalControl<TimeStepper>::calcDivT(const stdvec& control)
{
  assert(calculatedXi);
  // assumes Xi and Psi are calculated using the same control
  divT.clear();
  divT.reserve(N);
  
  for(size_t i = 0; i < N; i++)
  {
    auto Toverlap = overlapC( xi_t[i] , timeStepper.propagatorDeriv(control[i]) , psi_t[i] ) ;
    divT.push_back(Toverlap);
  }
}

template<class TimeStepper>
void OptimalControl<TimeStepper>::calcPsiXiDivT(const stdvec& control)
{
  // should be parallized
  std::thread psiThread(std::bind(&OptimalControl<TimeStepper>::calcPsi, this, control));
  std::thread xiThread(std::bind(&OptimalControl<TimeStepper>::calcXi, this, control));
  psiThread.join();
  xiThread.join();
  calcDivT(control);
}

template<class TimeStepper>
double OptimalControl<TimeStepper>::calcCost(const stdvec& control, const bool new_control)
{
  if (new_control)
  {
    calculatedXi = false;
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
 else {
   return basis.convertHessian(
                     calcHessian(basis.convertControl(control,new_control),new_control)
                               );
 }
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
