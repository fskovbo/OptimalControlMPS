#include "BH_nlp.hpp"
// NLP must be included first due to library clash with ITensor
#include "OptimalControl.hpp"
#include "ControlBasisFactory.hpp"
#include "SeedGenerator.hpp"
#include "IpIpoptApplication.hpp"
#include "itensor/all.h"
#include "BH_sites.h"
#include "correlations.hpp"
#include "BH_tDMRG.hpp"
#include "InitializeState.hpp"
#include <stdlib.h>
#include <time.h>
#include <string>

using namespace itensor;
using namespace Ipopt;


int main(int argc, char* argv[]){

  if(argc < 2) {
    printfln("Usage: %s InputFile_BHcontrol",argv[0]);
    return 0;
  }

  auto input      = InputGroup(argv[1],"input");

  double tstep    = input.getReal("tstep",1e-2);
  double T        = input.getReal("T");

  int N           = input.getInt("N");
  int Npart       = input.getInt("Npart");
  int locDim      = input.getInt("d");

  double J        = 1.0;
  double U_i      = 2.5;
  double U_f      = 50;

  int M              = input.getInt("M");
  double gamma       = input.getReal("gamma",0);
  bool cache         = input.getYesNo("cacheProgress",false);
  int maxBondDim     = input.getInt("maxBondDim",100);
  double optTol      = input.getReal("optTol",1e-7);
  double threshold   = input.getReal("threshold",1e-7);
  size_t threadCount = input.getInt("threadCount",2);
  int maxIter        = input.getInt("maxIter",200);
  double maxCPUHours = input.getReal("maxCPUHours",24);
  double ObjScaling  = input.getReal("ObjScaling",1);
  double maxCPUTime = maxCPUHours*60*60;

  
  int seed      = 1;

  if(argc > 2) seed = std::stoi(argv[2]);
  else printfln("Default seed used");

  srand ((unsigned) seed*time(NULL));


  std::cout << "Performing optimal control of Bose-Hubbard model ... \n\n";
  std::cout << " ******* Parameters used ******* \n";
  std::cout << "Number of sites ................ " << N << "\n";
  std::cout << "Number of particles ............ " << Npart << "\n";
  std::cout << "Local Fock space dimension ..... " << locDim << "\n";
  std::cout << "Control duration ............... " << T << "\n";
  std::cout << "Time-step size ................. " << tstep << "\n";
  std::cout << "GROUP dimension ................ " << M << "\n";
  std::cout << "Gamma (regularisation) ......... " << gamma << "\n";
  std::cout << "Maximum bond dimension (MPS).... " << maxBondDim << "\n";
  std::cout << "Truncation threshold (MPS) ..... " << threshold << "\n";
  std::cout << "Objective Scaling (IPOPT) ...... " << ObjScaling << "\n";
  std::cout << "Optimization tolerance (IPOPT).. " << optTol << "\n";
  std::cout << "MaxITER (IPOPT) ................ " << maxIter << "\n";
  std::cout << "MaxCPUTime (IPOPT) ............. " << maxCPUTime << "\n";
  std::cout << "ThreadCount......................" << threadCount << "\n";
  std::cout << "Seed  .......................... " << seed << "\n\n\n";


  auto sites    = BoseHubbard(N,locDim);
  auto u0       = SeedGenerator::linsigmoidSeed(U_i,U_f,T/tstep+1);
  auto basis    = ControlBasisFactory::buildChoppedSineBasis(u0,tstep,T,M);
  auto psi_i    = InitializeState(sites,Npart,J,u0.front(),maxBondDim,threshold);
  auto psi_f    = InitializeState(sites,Npart,J,u0.back(),maxBondDim,threshold);

  auto stepper  = BH_tDMRG(sites,J,tstep,{"Cutoff=",threshold,"Maxm=",maxBondDim});
  OptimalControl<BH_tDMRG> OC(psi_f,psi_i,stepper,basis,gamma);
  OC.setThreadCount(threadCount);

  // Create a new instance of your nlp
  //  (use a SmartPtr, not raw)
  SmartPtr<TNLP> mynlp = new BH_nlp(OC,cache);

  // Create a new instance of IpoptApplication
  //  (use a SmartPtr, not raw)
  // We are using the factory, since this allows us to compile this
  // example with an Ipopt Windows DLL
  SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

  // Change some options
  // Note: The following choices are only examples, they might not be
  //       suitable for your optimization problem.
  app->Options()->SetNumericValue("tol", optTol);
  app->Options()->SetStringValue("mu_strategy", "adaptive");
  app->Options()->SetStringValue("jac_d_constant","yes");
  app->Options()->SetIntegerValue("max_iter",maxIter);
  app->Options()->SetNumericValue("max_cpu_time",maxCPUTime);
  app->Options()->SetNumericValue("obj_scaling_factor",ObjScaling);
  // app->Options()->SetStringValue("hessian_approximation", "limited-memory");
  // app->Options()->SetStringValue("output_file", "logfile_BH.txt");
  // app->Options()->SetStringValue("derivative_test", "first-order");

  // Intialize the IpoptApplication and process the options
  ApplicationReturnStatus status;
  status = app->Initialize();
  if (status != Solve_Succeeded) {
    printf("\n\n*** Error during initialization!\n");
    return 0;
  }

  // Ask Ipopt to solve the problem
  status = app->OptimizeTNLP(mynlp);
  int returnstatus;

  if (status == Solve_Succeeded) {
    printf("\n\n*** The problem solved!\n");
    returnstatus = 1;
  }
  else {
    printf("\n\n*** The problem FAILED!\n");
    returnstatus = 0;
  }

  // As the SmartPtrs go out of scope, the reference count
  // will be decremented and the objects will automatically
  // be deleted.

  // Extract psi for each t, evaluate expectation value
  // of number operator, and save to file.
  auto psi_t = OC.getPsit();
  auto times = OC.getTimeAxis();
  std::string filename = "ExpectationN.txt";
  std::ofstream myfile (filename);
  if (myfile.is_open())
  {
    size_t ind = 0;
    for (auto& psi : psi_t){
      myfile << times.at(ind++) << "\t";
      auto expn = expectationValues(sites,psi,"N");
      for (auto& val : expn){
        myfile << val.real() << "\t";
      }
      myfile << "\n";
    }
    myfile.close();
  }
  else std::cout << "Unable to open file\n";
  // Calculate the final Hessian matrix
  return 0;
}
