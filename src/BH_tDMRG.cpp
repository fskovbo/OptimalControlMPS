#include "BH_tDMRG.hpp"

BH_tDMRG::BH_tDMRG(const SiteSet& sites, const double J, const double tstep, const Args& args)
  : J(J), sites(sites), args(args), d(sites.si(1).nblock())
{
  // builds J-Gates as well, which remain constant
  setTstep(tstep);

  // derivative of propagator is constant -> pre-build it 
  auto ampo = AutoMPO(sites);
  for(int i = 1; i <= sites.N(); ++i) {
    ampo += 0.5,"N(N-1)",i;
  }
  propDeriv = IQMPO(ampo);
}


void BH_tDMRG::initJGates(const double J)
{
  using Gate = BondGate<IQTensor>;

  // Even Gates first, then odd gates
  JGates_tforwards.clear();
  JGates_tbackwards.clear();

  // build J-Gates for even bonds
  // builds gates for both forwards and backwards propagation
  for(int i = 1; i < sites.N(); i += 2)
  {
    // get local J-Hamiltonian for sites i, i+1    
    auto hterm = -J*sites.op("A",i)*sites.op("Adag",i+1);
    hterm += -J*sites.op("Adag",i)*sites.op("A",i+1);

    // build gates for forward and backwards propagation
    auto gf = Gate(sites,i,i+1, Gate::tReal, tstep, hterm);
    auto gb = Gate(sites,i,i+1, Gate::tReal, -tstep, hterm);
    JGates_tforwards.push_back(gf);
    JGates_tbackwards.push_back(gb);
  }

  int offset = 1;
  if (sites.N() % 2 == 0) { offset = 2; }

  // build J-Gates for odd bonds
  // builds gates for both forwards and backwards propagation  
  for(int i = sites.N()-offset; i >= 1; i -= 2)
  {
    // get local J-Hamiltonian for sites i, i+1    
    auto hterm = -J*sites.op("A",i)*sites.op("Adag",i+1);
    hterm += -J*sites.op("Adag",i)*sites.op("A",i+1);

    // build gates for forward and backwards propagation
    auto gf = Gate(sites,i,i+1,Gate::tReal,tstep,hterm);
    auto gb = Gate(sites,i,i+1,Gate::tReal,-tstep,hterm);
    JGates_tforwards.push_back(gf);
    JGates_tbackwards.push_back(gb);
  }
}


void BH_tDMRG::setTstep(const double tstep_)
{
  tstep = tstep_;
  initJGates(J);
}


double BH_tDMRG::getTstep() const
{
  return tstep;
}


void BH_tDMRG::initUGates(UGatePair& UGates, const double Ufrom, const double Uto) const
{ 
  UGates.first.reserve(sites.N());
  UGates.second.reserve(sites.N());

  // prebuild diagonal entries of U-Gates
  std::vector<Cplx> expFrom, expTo;
  expFrom.reserve(d);
  expTo.reserve(d);
  for(size_t i = 0; i < d; i++)
  {
    expFrom.push_back( std::exp( -0.25*Ufrom*tstep*Cplx_i*i*(i-1) ) );
    expTo.push_back( std::exp( -0.25*Uto*tstep*Cplx_i*i*(i-1) ) );
  }
  
  // U-Gates are diagonal, whereby H_U can be exponentiated directly
  for (int k = 1; k <= sites.N(); ++k)
  {
    // two indices needed for U-Gates
    auto s    = sites.si(k);
    auto sP   = prime(s);

    IQTensor T1(dag(s),sP);
    auto T2 = T1;

    for (size_t i = 0; i < d; i++)
    {
      T1.set(s(i+1),sP(i+1), expFrom[i] );
      T2.set(s(i+1),sP(i+1), expTo[i] );
    }

    UGates.first.push_back(T1);
    UGates.second.push_back(T2);
  }
}


void BH_tDMRG::step(IQMPS& psi, const double from, const double to, const bool propagateForward) const
{
  // U-Gates are not member, as they are repeatedly changes,
  // which is a problem when multithreading
  UGatePair UGates;

  if (propagateForward) {
    initUGates(UGates, from, to);
    doStep(psi, UGates, JGates_tforwards);
  }
  else {
    initUGates(UGates, -from, -to);
    doStep(psi, UGates, JGates_tbackwards);
  }
}

void BH_tDMRG::doStep(IQMPS& psi, const UGatePair& UGates, const GateList& JGates) const
{
  auto& UGates1 = UGates.first;
  auto& UGates2 = UGates.second;

  // if N odd: "lonely" UGate at last site must be applied first
  if (sites.N() % 2 != 0) { // N is odd
    psi.Aref(sites.N()) *= UGates1.back();
    psi.Aref(sites.N()).mapprime(1,0,Site);
  }

  bool movingFromLeft = true;
  IQTensor AA;
  auto g = JGates.begin();
  while(g != JGates.end()) // iterating over all J-Gates (even gates first)
  {
    // get indices of Gate
    auto i1 = g->i1();
    auto i2 = g->i2();
    if (movingFromLeft)
    {
      // merge sites i1 and i2 into 2-site tensor AA
      // Apply U-gates first, then J-gate
      AA = psi.Aref(i1)*psi.Aref(i2)*UGates1[i1-1]*UGates1[i2-1]*prime(g->gate(),Site);

      // if N even, apply lonely Ugate at right side in the end of left move
      if (i2 == sites.N() && sites.N() % 2 == 0){ // N is even
        AA *= prime(prime(UGates2.back(),Site),Site);
      }
    } else { // moving from right-to-left
      // merge sites i1 and i2 into 2-site tensor AA
      // Apply J-gate first, then U-gates
      AA = psi.Aref(i1)*psi.Aref(i2)*g->gate()*prime(UGates2[i1-1],Site)*prime(UGates2[i2-1],Site);
    }

    // remove all prime levels, so next layer of network can be applied
    AA.noprime(Site);

    // prepare for next gate by splitting two-site tensor AA and moving central site of MPS
    ++g;
    if(g != JGates.end()) // there are more gates in this propagation step
    {
      // Look ahead to next gate position
      auto ni1 = g->i1();
      auto ni2 = g->i2();

      if(ni1 >= i2) // current and next step are moving left-to-right
      {
        // SVD AA to restore MPS form
        // denmatDecomp is slightly faster than standard SVD method from ITensor,
        // however one must set position of left and right canonical form of MPS manually
        denmatDecomp(AA,psi.Aref(i1),psi.Aref(i2),Fromleft,args);
        psi.leftLim(i1);
        if(psi.rightLim() < i1+2) psi.rightLim(i1+2);

        // normalize central site at position i1+1
        auto nrm = itensor::norm(psi.Aref(i1+1));
        if(nrm > 1E-16) psi.Aref(i1+1) *= 1./nrm;


        psi.position(ni1); //does no work if position already ni1
      }
      if(ni1 < i2) // current and next step are moving right-to-left
      {
        denmatDecomp(AA,psi.Aref(i1),psi.Aref(i2),Fromright,args);
        if(psi.leftLim() > i1-1) psi.leftLim(i1-1);
        psi.rightLim(i1+1);

        auto nrm = itensor::norm(psi.Aref(i1));
        if(nrm > 1E-16) psi.Aref(i1) *= 1./nrm;

        psi.position(ni2); //does no work if position already ni2
      }
      if (i2 == ni1 || i1 == ni2) // condition for odd N || condition for even N
      {
        // current step moved left-to-right, while next step is right-to-left
        movingFromLeft = false;
      }
    }
    else // there are no more gates in this propagation step
    {
      // No next gate to analyze, just restore MPS form through SVD
      denmatDecomp(AA,psi.Aref(i1),psi.Aref(i2),Fromright,args);
      psi.leftLim(i1-1);
      psi.rightLim(i1+1);

      auto nrm = itensor::norm(psi.Aref(i1));
      if(nrm > 1E-16) psi.Aref(i1) *= 1./nrm;

      // Gauge MPS to site 1, so it is ready for next propagation step
      psi.position(1);
    }
  }

  // "lonely" UGate at site 1 must be applied last
  psi.Aref(1) *= UGates2.front();
  psi.Aref(1).mapprime(1,0,Site);

  // Normalize in the end, to make sure that the MPS is truly normalized
  // This operation is very cheap, if the normalization is contained in a single site
  // i.e. the canonical form of the MPS has been maintained through all the steps above 
  psi.normalize();

}

Args BH_tDMRG::getArgs() const
{
  return args;
}


IQMPO BH_tDMRG::propagatorDeriv(const double& control_n) const
{
  return propDeriv;
}
