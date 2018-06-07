#include "BH_tDMRG.hpp"

BH_tDMRG::BH_tDMRG(const SiteSet& sites, const double J, const double tstep, const Args& args)
  : J(J), sites(sites), args(args)
{
  // builds J-Gates as well, which remain constant
  setTstep(tstep);

  // derivative of propagator is constant
  auto ampo = AutoMPO(sites);
  for(int i = 1; i <= sites.N(); ++i) {
    ampo += 0.5,"N(N-1)",i;
  }
  propDeriv = IQMPO(ampo);
}


void BH_tDMRG::initJGates(const double J)
{
  using Gate = BondGate<IQTensor>;

  JGates_tforwards.clear();
  JGates_tbackwards.clear();

  // build J-Gates for even bonds
  // builds gates for both forwards and backwards propagation
  for(int i = 1; i < sites.N(); i += 2)
      {
      auto hterm = -J*sites.op("A",i)*sites.op("Adag",i+1);
      hterm += -J*sites.op("Adag",i)*sites.op("A",i+1);

      auto gf = Gate(sites,i,i+1,Gate::tReal,tstep,hterm);
      auto gb = Gate(sites,i,i+1,Gate::tReal,-tstep,hterm);
      JGates_tforwards.push_back(gf);
      JGates_tbackwards.push_back(gb);
      }

  int offset = 1;
  if (sites.N() % 2 == 0) { offset = 2; }

  // build J-Gates for odd bonds
  // builds gates for both forwards and backwards propagation  
  for(int i = sites.N()-offset; i >= 1; i -= 2)
      {
      auto hterm = -J*sites.op("A",i)*sites.op("Adag",i+1);
      hterm += -J*sites.op("Adag",i)*sites.op("A",i+1);

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


void BH_tDMRG::initUGates(const double Ufrom, const double Uto)
{
  UGates1.clear();
  UGates2.clear();

  // U-Gates are diagonal, whereby H_U can be exponentiated directly
  for (int k = 1; k <= sites.N(); ++k) {
    auto s    = sites.si(k);
    auto sP   = prime(s);
    int HD    = s.nblock();

    IQTensor T1(dag(s),sP);
    auto T2 = T1;

    for (size_t i = 0; i < HD; i++) {
      T1.set(s(i+1),sP(i+1), std::exp( -0.25*Ufrom*tstep*Cplx_i*i*(i-1) ) );
      T2.set(s(i+1),sP(i+1), std::exp( -0.25*Uto*tstep*Cplx_i*i*(i-1) ) );
    }

    UGates1.push_back(T1);
    UGates2.push_back(T2);
  }
}


void BH_tDMRG::step(IQMPS& psi, const double from, const double to, const bool propagateForward)
{
  if (propagateForward) {
    initUGates(from,to);
    doStep(psi, JGates_tforwards);
  }
  else {
    initUGates(-from,-to);
    doStep(psi, JGates_tbackwards);
  }
}

void BH_tDMRG::doStep(IQMPS& psi, const GateList& JGates)
{
  // if N odd: "lonely" UGate at end must be applied first
  if (sites.N() % 2 != 0) { // N is odd
    psi.Aref(sites.N()) *= UGates1.back();
    psi.Aref(sites.N()).mapprime(1,0,Site);
  }

  bool movingFromLeft = true;
  IQTensor AA;
  auto g = JGates.begin();
  while(g != JGates.end())
      {
      auto i1 = g->i1();
      auto i2 = g->i2();
      if (movingFromLeft) {
        AA = psi.Aref(i1)*psi.Aref(i2)*UGates1[i1-1]*UGates1[i2-1]*prime(g->gate(),Site);

        // if N even apply lonely Ugate at right side in the end of left move
        if (i2 == sites.N() && sites.N() % 2 == 0) { // N is even
          AA *= prime(prime(UGates2.back(),Site),Site);
        }
      } else {
        AA = psi.Aref(i1)*psi.Aref(i2)*g->gate()*prime(UGates2[i1-1],Site)*prime(UGates2[i2-1],Site);
      }

      AA.noprime(Site);

      ++g;
      if(g != JGates.end())
          {
          //Look ahead to next gate position
          auto ni1 = g->i1();
          auto ni2 = g->i2();
          //SVD AA to restore MPS form
          //before applying current gate
          if(ni1 >= i2)
              {
              denmatDecomp(AA,psi.Aref(i1),psi.Aref(i2),Fromleft,args);
              psi.leftLim(i1);
              if(psi.rightLim() < i1+2) psi.rightLim(i1+2);

              auto nrm = itensor::norm(psi.Aref(i1+1));
              if(nrm > 1E-16) psi.Aref(i1+1) *= 1./nrm;

              psi.position(ni1); //does no work if position already ni1
              }
          if(ni1 < i2)
              {
              denmatDecomp(AA,psi.Aref(i1),psi.Aref(i2),Fromright,args);
              if(psi.leftLim() > i1-1) psi.leftLim(i1-1);
              psi.rightLim(i1+1);

              auto nrm = itensor::norm(psi.Aref(i1));
              if(nrm > 1E-16) psi.Aref(i1) *= 1./nrm;

              psi.position(ni2); //does no work if position already ni2
              }
          if (i2 == ni1 || i1 == ni2) // odd condition || even condition
              {
              movingFromLeft = false;
              }
          }
      else
          {
          //No next gate to analyze, just restore MPS form
          denmatDecomp(AA,psi.Aref(i1),psi.Aref(i2),Fromright,args);
          psi.leftLim(i1-1);
          psi.rightLim(i1+1);

          auto nrm = itensor::norm(psi.Aref(i1));
          if(nrm > 1E-16) psi.Aref(i1) *= 1./nrm;

          psi.position(1);
          }
      }

  // "lonely" UGate at start must be applied last
  psi.Aref(1) *= UGates2.front();
  psi.Aref(1).mapprime(1,0,Site);

  psi.normalize();

}

Args BH_tDMRG::getArgs() const{
    return args;
}


IQMPO BH_tDMRG::propagatorDeriv(const double& control_n)
{
  return propDeriv;
}
