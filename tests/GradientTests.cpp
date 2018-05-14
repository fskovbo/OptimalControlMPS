#include <gtest/gtest.h>
#include "itensor/all.h"

#include "../include/BH_sites.h"
#include "../include/BH_tDMRG.hpp"
#include "../include/OptimalControl.hpp"
#include "../include/InitializeState.hpp"



using namespace itensor;


struct gradientTest : testing::Test
{    
    SiteSet* sites;
    IQMPS* psi_i;
    IQMPS* psi_f;
    BH_tDMRG* tDMRG;

    double cstart;
    double cend;
    double T;
    double tstep;
    
    gradientTest()
    {
        int N         = 5;
        int Npart     = 5;
        int locDim    = 5;

        double J      = 1.0;
        cstart        = 3.0;
        cend         = 10.0;
        T            = 0.5;
        tstep        = 1e-2;

        *sites    = BoseHubbard(N,locDim);
        *psi_i        = InitializeState(*sites,Npart,J,cstart);
        *psi_f        = InitializeState(*sites,Npart,J,cend);

        *tDMRG        = BH_tDMRG(*sites,J,tstep,{"Cutoff=",1E-8});
    }
    ~gradientTest()
    {
        delete sites;
        delete psi_i;
        delete psi_f;
        delete tDMRG;
    }

    std::vector<double> linseed()
    {
        std::vector<double> array;
        double n = T/(tstep) + 1;
        double a = cstart;
        double b = cend;
        double step = (b-a) / (n-1);

        while(a <= b + 1e-7) {
            array.push_back(a);
            a += step;           // could recode to better handle rounding errors
        }
        return array;
    }

    std::vector<double> sinseed()
    {
        auto lin = linseed();
        
        for (auto& val : lin){
            val = sin(val);
        }
        
        return lin;
    }

    std::vector<double> expseed()
    {
        auto lin = linseed();
        
        for (auto& val : lin){
            val = exp(val);
        }
        
        return lin;
    }

    std::vector<double> getNumericRegGrad(std::vector<double> control,OptimalControl<BH_tDMRG>& OCBH)
    {
        double Jp, Jm, epsilon = 1e-5;
        std::vector<double> g;
        g.reserve(control.size());

        size_t count = 0;

        for (auto& ui : control){
            ui        += epsilon;
            Jp         = OCBH.getRegularization(control);

            ui        -= 2.0*epsilon;
            Jm         = OCBH.getRegularization(control);

            ui        += epsilon;
            g.push_back((Jp-Jm)/(2.0*epsilon));
        }
        
        return g;
    }



};

TEST_F(gradientTest, testRegularization)
{
    OptimalControl<BH_tDMRG> OC(*psi_f,*psi_i,*tDMRG,1e-4);

    auto lincontrol = linseed();
    auto analytic   = OC.getRegularizationGrad(lincontrol);
    auto numeric    = getNumericRegGrad(lincontrol,OC);

    ASSERT_EQ( analytic.size() , numeric.size() ); 
    
    for(size_t i = 1; i < analytic.size()-1; i++)
    {
        EXPECT_NEAR(analytic.at(i) , numeric.at(i), 1e-5);
    }
    
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}