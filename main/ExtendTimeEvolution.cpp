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

    if(argc < 3) {
        printfln("Usage: %s InputFile_BHcontrol BHrampInitialFinal.txt",argv[0]);
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

    int M           = input.getInt("M");
    double gamma    = input.getReal("gamma",0);
    bool cache      = input.getYesNo("cacheProgress",false);
    int maxBondDim  = input.getInt("maxBondDim",100);
    double optTol   = input.getReal("optTol",1e-7);
    double threshold= input.getReal("threshold",1e-7);
    int seed        = 1;

    // load BHrampInitialFinal.txt
    std::string RampDataFile(argv[2]);
    std::vector<double> control_final, control_init, times;
    double tmp, t, u_i, u_f;
    std::ifstream in(RampDataFile);
    while(!in.eof())
    {
        in >> t >> u_i >> tmp >> u_f >> tmp;
        times.push_back(t);
        control_init.push_back(u_i);
        control_final.push_back(u_f);
    }


    // extend duration and controls
    for (size_t i = 1; i <= 100; i++) {
        times.push_back(T + i*tstep);
        control_init.push_back( control_init.back() );
        control_final.push_back( control_final.back() );
    }

    auto sites    = BoseHubbard(N,locDim);
    auto basis    = ControlBasisFactory::buildChoppedSineBasis(u0,tstep,T,M);
    auto psi_i    = InitializeState(sites,Npart,J,u0.front(),maxBondDim,threshold);
    auto psi_f    = InitializeState(sites,Npart,J,u0.back(),maxBondDim,threshold);

    auto stepper  = BH_tDMRG(sites,J,tstep,{"Cutoff=",threshold,"Maxm=",maxBondDim});
    OptimalControl<BH_tDMRG> OC(psi_f,psi_i,stepper,times.size(),gamma);


    // calculate extended fidelities
    auto fid_init   = OC.getFidelityForAllT(control_init);
    std::cout << "Calculated initial control" << '\n';
    auto fid_final  = OC.getFidelityForAllT(control_final);
    std::cout << "Calculated final control" << '\n';


    // Save extended Data

    // Create an output string stream
    std::ostringstream streamObj;

    // Set Fixed -Point Notation
    streamObj << std::fixed;

    // Set precision to 1 digits
    streamObj << std::setprecision(1);

    //Add double to stream
    streamObj << T;

    std::string filename1 = "BHrampInitialFinal_extendedT" + streamObj.str() + ".txt";
    std::ofstream myfile1 (filename1);
    if (myfile1.is_open())
    {
        for (int i = 0; i < times.size(); i++) {
        myfile1 << times.at(i)        << "\t";
        myfile1 << control_init.at(i) << "\t";
        myfile1 << fid_init.at(i)     << "\t";
        myfile1 << control_final.at(i)<< "\t";
        myfile1 << fid_final.at(i)    << "\n";
        }
        myfile1.close();
    }
    else std::cout << "Unable to open file\n";
    std::cout << "Saved ramp data" << '\n';


    // Extract psi for each t, evaluate expectation value
    // of number operator, and save to file.
    auto psi_t = OC.getPsit();
    std::string filename2 = "ExpectationN_extendedT" + streamObj.str() + ".txt";
    std::ofstream myfile2 (filename2);
    if (myfile2.is_open())
    {
        size_t ind = 0;
        for (auto& psi : psi_t)
        {
            myfile2 << times.at(ind++) << "\t";
            auto expn = expectationValues(sites,psi,"N");
            for (auto& val : expn)
            {
                myfile2 << val.real() << "\t";
            }
            myfile2 << "\n";
        }
        myfile2.close();
    }
    else std::cout << "Unable to open file\n";
    std::cout << "Saved population data" << '\n';

    return 0;
}
