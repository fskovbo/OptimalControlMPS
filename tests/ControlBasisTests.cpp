#include <gtest/gtest.h>
#include "itensor/all.h"

#include "../include/ControlBasis.hpp"

using rowmat = std::vector< std::vector<double>>;
#define PI 3.14159265

struct simpleMatrixTest : testing::Test {

    *ControlBasis basis;

    twoSiteCorrelationTest()
    {
        int N           = 5;
        int M           = 4;
        double T        = 2;
        double tstep    = 1e-1;

        std::vector<double> u0(N, 1.0);
        std::vector<double> S(N, 1.0);


        rowmat f(N, std::vector(M, 2.0));
        
        /* for(size_t i = 0; i < N; i++)
        {
            for(size_t n = 0; n < M; n++)
            {
                f[i][n] = sin( (n+1)*PI*tstep*i/T);
            }   
        } */
        basis = new ControlBasis(u0,S,f);
    }

    ~twoSiteCorrelationTest()
    {
        delete basis;
    }

};


TEST_F(simpleMatrixTest, testConvertControl)
{
    // test c = {0 , 0 , ...}
    std::vector<double> c1(M,0);
    auto u1 = basis.convertControl(c1);

    ASSERT_EQ( u1.size() , N ); 
    for(size_t i = 0; i < N; i++)
    {
        EXPECT_NEAR(u1.at(i) , 1.0 , 1e-8);
    }

    // test c = {1 , 1 , ...}
    std::vector<double> c2(M,1);
    auto u2 = basis.convertControl(c2);

    ASSERT_EQ( u2.size() , N ); 
    for(size_t i = 0; i < N; i++)
    {
        EXPECT_NEAR(u2.at(i) , 1+2.0*M , 1e-8);
    }

    // test new_control
    auto u3 = basis.convertControl(c1,false);
    for(size_t i = 0; i < N; i++)
    {
        EXPECT_NEAR(u2.at(i) , u3.at(i) , 1e-8);
    }
}


TEST_F(simpleMatrixTest, testConvertGradient)
{
    // test gradu = {0 , 0 , ...}
    std::vector<double> gradu1(N,0);
    auto gradc1 = basis.convertGradient(gradu1);

    ASSERT_EQ( gradc1.size() , M ); 
    for(size_t i = 0; i < M; i++)
    {
        EXPECT_NEAR(gradc1.at(i) , 0 , 1e-8);
    }

    // test gradu = {1 , 1 , ...}
    std::vector<double> gradu2(N,0);
    auto gradc2 = basis.convertGradient(gradu2);

    ASSERT_EQ( gradc2.size() , M ); 
    for(size_t i = 0; i < M; i++)
    {
        EXPECT_NEAR(gradc2.at(i) , 2.0*N , 1e-8);
    }

} 


TEST_F(simpleMatrixTest, testControlJacobian)
{
    auto jac = basis.getControlJacobian();

    ASSERT_EQ( jac.size() , N );
    ASSERT_EQ( jac.front().size() , M );

    for(size_t i = 0; i < N; i++)
    {
        for(size_t j = 0; j < M; j++)
        {
            EXPECT_NEAR(jac[i][j] , 2.0 , 1e-8);
        }
    }
} 



int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
