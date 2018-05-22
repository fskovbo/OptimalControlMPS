#include "itensor/all.h"
#include "BH_sites.h"
#include "InitializeState.hpp"
#include "BH_tDMRG.hpp"
#include "OptimalControl.hpp"
#include "SeedGenerator.hpp"
#include "correlations.hpp"
#include <string>
#include <iomanip>


using namespace itensor;


int main(int argc, char* argv[])
{
    // load InputFile_BHcontrol.txt
    auto input    = InputGroup(argv[1],"input");

    double tstep  = input.getReal("tstep",1e-2);
    double T      = input.getReal("T",6);

    int N         = input.getInt("N",8);
    int Npart     = input.getInt("Npart",8);
    int locDim    = input.getInt("d",8);

    double J      = 1.0;
    double U_i    = 2.0;
    double U_f    = 50;


    // load BHrampInitialFinal.txt
    std::string RampDataFile(argv[2]);
    std::vector<double> control, times;
    double tmp, t, u;
    std::ifstream in(RampDataFile);
    while(!in.eof())
    {
        in >> t >> tmp >> tmp >> u >> tmp;
        times.push_back(t);
        control.push_back(u);
    }

    auto sites    = BoseHubbard(N,locDim);
    auto psi_i    = InitializeState(sites,Npart,J,control.front());
    auto psi_f    = InitializeState(sites,Npart,J,control.back());

    auto stepper    = BH_tDMRG(sites,J,tstep,{"Cutoff=",1E-8});
    OptimalControl<BH_tDMRG> OC(psi_f,psi_i,stepper,0);


    // extend duration and controls
    for (size_t i = 1; i <= 50; i++) {
    times.push_back(T + i*tstep);
    control.push_back( control.back() );
    }

    
    // run time evolution and get fidelity
    auto fid        = OC.getFidelityForAllT(control);
    auto psi        = OC.getPsit();
    

    // Save extended Data

    // Create an output string stream
    std::ostringstream streamObj;

    // Set Fixed -Point Notation
    streamObj << std::fixed;

    // Set precision to 1 digits
    streamObj << std::setprecision(1);

    //Add double to stream
    streamObj << T;

    std::string filename = "TimeEvolBondDimT" + streamObj.str() + ".txt";
    std::ofstream myfile (filename);
    size_t ind = 0;

    if (myfile.is_open())
    {
        for (auto& pt : psi)
        {
            myfile << times.at(ind)     << "\t";
            myfile << control.at(ind)   << "\t";
            myfile << fid.at(ind)       << "\t";
            
            for(int b = 1; b < pt.N(); ++b) 
            {
                myfile << linkInd(pt,b).m() << "\t"; 
            }

            ind++;
            myfile << "\n";
        }
        myfile.close();
    }
    else std::cout << "Unable to open file\n";


    
    return 0;
}
