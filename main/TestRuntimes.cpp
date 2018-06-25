#include "itensor/all.h"
#include "BH_sites.h"
#include "InitializeState.hpp"
#include "BH_tDMRG.hpp"
#include "OptimalControl.hpp"
#include "SeedGenerator.hpp"
#include "correlations.hpp"
#include <string>
#include <iomanip>
#include <time.h>


using namespace itensor;


int main()
{
    std::vector<int> durations = {2 ,3 , 4, 5 , 6 , 7 ,8 ,9, 10 };

    double tstep    = 1e-2;

    int N           = 20;
    int Npart       = 20;
    int locDim      = 6;
   
    double J        = 1.0;
    double U_i      = 2.5;
    double U_f      = 50;

    auto sites      = BoseHubbard(N,locDim);
    auto psi_i      = InitializeState(sites,Npart,J,U_i);
    auto psi_f      = InitializeState(sites,Npart,J,U_f);
    auto stepper    = BH_tDMRG(sites,J,tstep,{"Cutoff=",1E-8,"Maxm=",40});

    std::vector<double> runtimes, costs;
    for (double T : durations)
    {
        std::cout << "Calculating time-evolution for T = " << T << "\n";

        auto control    = SeedGenerator::adiabaticSeed(U_i,U_f,T/tstep+1);
        OptimalControl<BH_tDMRG> OC(psi_f,psi_i,stepper,control.size(),0);
        
        // run time evolution and get fidelity
        clock_t begin   = clock();
        auto cost       = OC.getCost(control);
        clock_t end     = clock();

        double rt = double(end - begin) / CLOCKS_PER_SEC;
        runtimes.push_back(rt);    
        costs.push_back(cost);    
    } 

    
    for(size_t i = 0; i < durations.size(); i++)
    {
        std::cout << durations.at(i) << "\t";
        std::cout << runtimes.at(i) << "\t";
        std::cout << costs.at(i) << "\n";
    }
    
    return 0;
}
