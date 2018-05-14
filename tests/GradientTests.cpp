#include <gtest/gtest.h>
#include "itensor/all.h"

#include "../include/BH_sites.h"
#include "../include/BH_tDMRG.hpp"
#include "../include/OptimalControl.hpp"
#include "../include/InitializeState.hpp"
#include "../include/ControlBasisFactory.hpp"


using namespace itensor;


struct gradientTest : testing::Test
{    
    int N;
    int Npart;
    int locDim;

    double J;
    double cstart;
    double cend;
    double T;
    double tstep;
    
    gradientTest()
    {
        N           = 5;
        Npart       = 5;
        locDim      = 5;

        J           = 1.0;
        cstart      = 3.0;
        cend        = 10.0;
        T           = 0.5;
        tstep       = 1e-2;
    }
    ~gradientTest()
    {
    }

    std::vector<double> linseed(double a, double b, int n)
    {
        std::vector<double> array;
        double step = (b-a) / (n-1);

        while(a <= b + 1e-7) {
            array.push_back(a);
            a += step;           // could recode to better handle rounding errors
        }
        return array;
    }

    std::vector<double> sinseed(double a, double b, int n)
    {
        auto lin = linseed(a,b,n);
        
        for (auto& val : lin){
            val = sin(val);
        }
        
        return lin;
    }

    std::vector<double> expseed(double a, double b, int n)
    {
        auto lin = linseed(a,b,n);
        
        for (auto& val : lin){
            val = exp(val);
        }
        
        return lin;
    }

    std::vector<double> getNumericRegGrad(std::vector<double> control, OptimalControl<BH_tDMRG>& OCBH)
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

    std::vector<double> getNumericGrad(std::vector<double> control, OptimalControl<BH_tDMRG>& OCBH)
    {
        double Jp, Jm, epsilon = 1e-5;
        std::vector<double> g;
        g.reserve(control.size());

        size_t count = 0;

        for (auto& ui : control){
            ui        += epsilon;
            Jp         = OCBH.getCost(control);

            ui        -= 2.0*epsilon;
            Jm         = OCBH.getCost(control);

            ui        += epsilon;
            g.push_back((Jp-Jm)/(2.0*epsilon));
        }
        
        return g;
    }

    



};

TEST_F(gradientTest, testRegularizationGRAPE)
{
    auto sites      = BoseHubbard(N,locDim);
    auto psi_i      = InitializeState(sites,Npart,J,cstart);
    auto psi_f      = InitializeState(sites,Npart,J,cend);
    auto tDMRG      = BH_tDMRG(sites,J,tstep,{"Cutoff=",1E-8});
    OptimalControl<BH_tDMRG> OC(psi_f,psi_i,tDMRG,1e-4);

    // Test GRAPE regularization gradient for linear control
    auto lincontrol = linseed(cstart,cend,T/tstep);
    auto analytic   = OC.getRegularizationGrad(lincontrol);
    auto numeric    = getNumericRegGrad(lincontrol,OC);

    ASSERT_EQ( analytic.size() , numeric.size() ); 
    
    for(size_t i = 1; i < analytic.size()-1; i++)
    {
        EXPECT_NEAR(analytic.at(i) , numeric.at(i), 1e-5);
    }

    // Test GRAPE regularization gradient for sinus gradient
    lincontrol = sinseed(cstart,cend,T/tstep);
    analytic   = OC.getRegularizationGrad(lincontrol);
    numeric    = getNumericRegGrad(lincontrol,OC);

    ASSERT_EQ( analytic.size() , numeric.size() ); 
    
    for(size_t i = 1; i < analytic.size()-1; i++)
    {
        EXPECT_NEAR(analytic.at(i) , numeric.at(i), 1e-5);
    }

    // Test GRAPE regularization gradient for exponential gradient
    lincontrol = expseed(cstart,cend,T/tstep);
    analytic   = OC.getRegularizationGrad(lincontrol);
    numeric    = getNumericRegGrad(lincontrol,OC);

    ASSERT_EQ( analytic.size() , numeric.size() ); 
    
    for(size_t i = 1; i < analytic.size()-1; i++)
    {
        EXPECT_NEAR(analytic.at(i) , numeric.at(i), 1e-5);
    }
    
}

TEST_F(gradientTest, testFidelity)
{
    auto sites      = BoseHubbard(N,locDim);
    auto psi_i      = InitializeState(sites,Npart,J,cstart);
    auto psi_f      = InitializeState(sites,Npart,J,cend);
    auto tDMRG      = BH_tDMRG(sites,J,tstep,{"Cutoff=",1E-8});
    OptimalControl<BH_tDMRG> OC(psi_f,psi_i,tDMRG,0);

    // Test GRAPE fidelity gradient for linear control
    auto lincontrol = linseed(cstart,cend,T/tstep);
    auto analytic   = OC.getFidelityGrad(lincontrol);
    auto numeric    = getNumericGrad(lincontrol,OC);

    ASSERT_EQ( analytic.size() , numeric.size() ); 
    
    for(size_t i = 1; i < analytic.size()-1; i++)
    {
        EXPECT_NEAR(analytic.at(i) , numeric.at(i), 1e-5);
    }

    // Test GRAPE fidelity gradient for sinus gradient
    lincontrol = sinseed(cstart,cend,T/tstep);
    analytic   = OC.getFidelityGrad(lincontrol);
    numeric    = getNumericGrad(lincontrol,OC);

    ASSERT_EQ( analytic.size() , numeric.size() ); 
    
    for(size_t i = 1; i < analytic.size()-1; i++)
    {
        EXPECT_NEAR(analytic.at(i) , numeric.at(i), 1e-5);
    }

    // Test GRAPE fidelity gradient for exponential gradient
    lincontrol = expseed(cstart,cend,T/tstep);
    analytic   = OC.getFidelityGrad(lincontrol);
    numeric    = getNumericGrad(lincontrol,OC);

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