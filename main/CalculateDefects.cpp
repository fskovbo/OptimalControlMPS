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


int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printfln("Usage: %s InputFile_BHcontrol BHrampInitialFinal.txt",argv[0]);
        printfln("No input detected ... using standard parameters");
    }

    double tstep  = 1e-2;
    double T      = 5;

    int N         = 5;
    int Npart     = 5;
    int locDim    = 5;

    // load InputFile_BHcontrol.txt
    if (argc >= 2)
    {
        auto input  = InputGroup(argv[1],"input");
        tstep       = input.getReal("tstep",1e-2);
        T           = input.getReal("T",6);

        N           = input.getInt("N",8);
        Npart       = input.getInt("Npart",8);
        locDim      = input.getInt("d",8);    
    }


    double J      = 1.0;
    double U_i    = 2.5;
    double U_f    = 50;


    // load BHrampInitialFinal.txt
    std::vector<double> control, times;
    if (argc >= 3)
    {
        std::string RampDataFile(argv[2]);
        double tmp, t, u;
        std::ifstream in(RampDataFile);
        while(!in.eof())
        {
            in >> t >> tmp >> tmp >> u >> tmp;
            times.push_back(t);
            control.push_back(u);
        }
    } 
    else
    {
        times   = SeedGenerator::generateRange(0,tstep,T);
        control = SeedGenerator::adiabaticSeed(U_i,U_f,times.size());
    }

    auto sites      = BoseHubbard(N,locDim);
    auto psi_i      = InitializeState(sites,Npart,J,control.front());
    auto psi_f      = InitializeState(sites,Npart,J,control.back());

    auto stepper    = BH_tDMRG(sites,J,tstep,{"Cutoff=",1E-8,"Maxm=",70});
    OptimalControl<BH_tDMRG> OC(psi_f,psi_i,stepper,control.size(),0);
    
    // run time evolution and get fidelity
    auto fids       = OC.getFidelityForAllT(control);
    auto psi_t      = OC.getPsit();


    auto expn_i     = expectationValues(sites,psi_i,"N");
    auto expnn_i    = expectationValues(sites,psi_i,"NN");
    std::vector<double> F2_i;
    for(size_t i = 0; i < N; i++)
    {
        F2_i.push_back( expnn_i.at(i).real() - expn_i.at(i).real()*expn_i.at(i).real() );
    }

    std::vector<double> rho, F2;
    for (auto& psi : psi_t)
    {
        auto expn   = expectationValues(sites,psi,"N");
        auto expnn  = expectationValues(sites,psi,"NN");

        double rhoval = 0;
        double F2val = 0;
        for(size_t i = 0; i < N; i++)
        {
            rhoval  += fabs(expn.at(i).real() - 1.0);
            F2val   += (expnn.at(i).real() - expn.at(i).real()*expn.at(i).real())/F2_i.at(i);
        }
        rho.push_back(rhoval/N);
        F2.push_back(F2val/N);
    }

    
    for(size_t i = 0; i < times.size(); i++)
    {
        std::cout << times.at(i) << "\t";
        std::cout << fids.at(i) << "\t";
        std::cout << rho.at(i) << "\t";
        std::cout << F2.at(i) << "\n";
    }
    
    
    return 0;
}
