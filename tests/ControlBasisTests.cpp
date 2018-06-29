#include <gtest/gtest.h>
#include "itensor/all.h"

#include "../include/ControlBasis.hpp"
#include "../include/ControlBasisFactory.hpp"

using rowmat = std::vector< std::vector<double>>;
#define PI 3.14159265

struct SimpleMatrixTest : testing::Test {

    ControlBasis* basis;

    SimpleMatrixTest()
    {
        int N           = 5;
        int M           = 4;

        std::vector<double> u0(N, 1.0);
        std::vector<double> S(N, 1.0);

        rowmat f(N, std::vector<double>(M, 2.0));
        
        basis = new ControlBasis(u0,S,f);
    }

    ~SimpleMatrixTest()
    {
        delete basis;
    }

};


struct ChoppedSineTest : testing::Test {

    ControlBasis* basis;

    ChoppedSineTest()
    {
        double T        = 1;
        double tstep    = 1e-1;
        int N           = T/tstep + 1;
        int M           = 5;

        std::vector<double> u0 = { 1 , 1.1 , 1.2 , 1.3 , 1.4 , 1.5 , 1.6 , 1.7 , 1.8 , 1.9 , 2};
        auto basis1  = ControlBasisFactory::buildChoppedSineBasis(u0,tstep,T,M);
        basis = new ControlBasis(basis1);
    }

    ~ChoppedSineTest()
    {
        delete basis;
    }

};


TEST_F(SimpleMatrixTest, testConvertControl)
{
    // test c = {0 , 0 , ...}
    std::vector<double> c1(basis->getM() , 0);
    auto u1 = basis->convertControl(c1);

    ASSERT_EQ( u1.size() , basis->getN() ); 
    for(size_t i = 0; i < basis->getN(); i++)
    {
        EXPECT_NEAR(u1.at(i) , 1.0 , 1e-8);
    }

    // test c = {1 , 1 , ...}
    std::vector<double> c2(basis->getM(),1);
    auto u2 = basis->convertControl(c2);

    ASSERT_EQ( u2.size() , basis->getN() ); 
    for(size_t i = 0; i < basis->getN(); i++)
    {
        EXPECT_NEAR(u2.at(i) , 1+2.0*basis->getM() , 1e-8);
    }

    // test new_control
    auto u3 = basis->convertControl(c1,false);
    for(size_t i = 0; i < basis->getN(); i++)
    {
        EXPECT_NEAR(u2.at(i) , u3.at(i) , 1e-8);
    }
}


TEST_F(SimpleMatrixTest, testConvertGradient)
{
    // test gradu = {0 , 0 , ...}
    std::vector<double> gradu1(basis->getN(),0);
    auto gradc1 = basis->convertGradient(gradu1);

    ASSERT_EQ( gradc1.size() , basis->getM() ); 
    for(size_t i = 0; i < basis->getM(); i++)
    {
        EXPECT_NEAR(gradc1.at(i) , 0 , 1e-8);
    }

    // test gradu = {1 , 1 , ...}
    std::vector<double> gradu2(basis->getN(),1);
    auto gradc2 = basis->convertGradient(gradu2);

    ASSERT_EQ( gradc2.size() , basis->getM() ); 
    for(size_t i = 0; i < basis->getM(); i++)
    {
        EXPECT_NEAR(gradc2.at(i) , 2.0*basis->getN() , 1e-8);
    }

} 


TEST_F(SimpleMatrixTest, testControlJacobian)
{
    auto jac = basis->getControlJacobian();

    ASSERT_EQ( jac.size() , basis->getN() );
    ASSERT_EQ( jac.front().size() , basis->getM() );

    for(size_t i = 0; i < basis->getN(); i++)
    {
        for(size_t j = 0; j < basis->getM(); j++)
        {
            EXPECT_NEAR(jac[i][j] , 2.0 , 1e-8);
        }
    }
} 


TEST_F(SimpleMatrixTest, testConvertHessian)
{
    // test Hessu = zeros
    rowmat Hessu1(basis->getN(), std::vector<double>(basis->getN(), 0));
    auto Hessc1 = basis->convertHessian(Hessu1);

    ASSERT_EQ( Hessc1.size() , basis->getM() );
    ASSERT_EQ( Hessc1.front().size() , basis->getM() );  
    for(size_t i = 0; i < basis->getM(); i++)
    {
        for(size_t j = 0; j < basis->getM(); j++)
        {
            EXPECT_NEAR(Hessc1[i][j] , 0.0 , 1e-8);
        }
        
    }

    // test Hessu = ones
    rowmat Hessu2(basis->getN(), std::vector<double>(basis->getN(), 1));
    auto Hessc2 = basis->convertHessian(Hessu2);

    ASSERT_EQ( Hessc2.size() , basis->getM() );
    ASSERT_EQ( Hessc2.front().size() , basis->getM() );  
    for(size_t i = 0; i < basis->getM(); i++)
    {
        for(size_t j = 0; j < basis->getM(); j++)
        {
            EXPECT_NEAR(Hessc2[i][j] , (basis->getN()*2.0)*(basis->getN()*2.0) , 1e-8);
        }
        
    }

    // test Hessu = identity
    rowmat Hessu3(basis->getN(), std::vector<double>(basis->getN(), 0));
    for(size_t i = 0; i < basis->getN(); i++)
    {
        Hessu3[i][i] = 1;
    }
    
    auto Hessc3 = basis->convertHessian(Hessu3);

    ASSERT_EQ( Hessc3.size() , basis->getM() );
    ASSERT_EQ( Hessc3.front().size() , basis->getM() );  
    for(size_t i = 0; i < basis->getM(); i++)
    {
        for(size_t j = 0; j < basis->getM(); j++)
        {
            EXPECT_NEAR(Hessc3[i][j] , basis->getN()*4.0 , 1e-8);
        }
        
    }

} 


TEST_F(ChoppedSineTest, testConvertControl)
{
    // test c = {0 , 0 , ...}
    std::vector<double> c1(basis->getM() , 0);
    auto u1 = basis->convertControl(c1);

    ASSERT_EQ( u1.size() , basis->getN() );
    for(size_t i = 0; i < basis->getN(); i++)
    {
        EXPECT_NEAR(u1.at(i) , 1+i*0.1 , 1e-6);
    }

    // test c = {1 , 1 , ...}
    std::vector<double> c2(basis->getM(),1);
    auto u2 = basis->convertControl(c2);

    // expected results calculated using old version of program
    std::vector<double> res2 = {1 ,4.75688 ,4.27768 ,1.78131 ,1.4 ,2.5 ,2.32654 ,1.45476, 1.8, 2.47919, 2 };
    ASSERT_EQ( u2.size() , basis->getN() ); 
    for(size_t i = 0; i < basis->getN(); i++)
    {
        EXPECT_NEAR(u2.at(i) , res2.at(i) , 5e-6);
    }

    // test new_control
    auto u3 = basis->convertControl(c1,false);
    for(size_t i = 0; i < basis->getN(); i++)
    {
        EXPECT_NEAR(u2.at(i) , u3.at(i) , 5e-6);
    }
}


TEST_F(ChoppedSineTest, testConvertGradient)
{
    // test gradu = {0 , 0 , ...}
    std::vector<double> gradu1(basis->getN(),0);
    auto gradc1 = basis->convertGradient(gradu1);

    ASSERT_EQ( gradc1.size() , basis->getM() ); 
    for(size_t i = 0; i < basis->getM(); i++)
    {
        EXPECT_NEAR(gradc1.at(i) , 0 , 5e-6);
    }

    // test gradu = {1 , 1 , ...}
    std::vector<double> gradu2(basis->getN(),1);
    auto gradc2 = basis->convertGradient(gradu2);

    // expected results calculated using old version of program
    std::vector<double> res2 = {6.31375 , 3.58979e-09 , 1.96261 , 7.17958e-09 , 1};
    ASSERT_EQ( gradc2.size() , basis->getM() ); 
    for(size_t i = 0; i < basis->getM(); i++)
    {
        EXPECT_NEAR(gradc2.at(i) , res2.at(i) , 5e-6);
    }

} 


TEST_F(ChoppedSineTest, testControlJacobian)
{
    // expected results calculated using old version of program
    std::vector< std::vector<double>> testJac;
    
    testJac.emplace_back(std::initializer_list<double>{ 0	,0	,0,	0,	0	});
    testJac.emplace_back(std::initializer_list<double>{ 0.309017,	0.587785,	0.809017,	0.951057,	1	});
    testJac.emplace_back(std::initializer_list<double>{ 0.587785,	0.951057,	0.951057,	0.587785,	3.58979e-09	});
    testJac.emplace_back(std::initializer_list<double>{ 0.809017,	0.951057,	0.309017,	-0.587785,	-1		});
    testJac.emplace_back(std::initializer_list<double>{ 0.951057,	0.587785,	-0.587785,	-0.951057,	-7.17959e-09,	});
    testJac.emplace_back(std::initializer_list<double>{ 1,	3.58979e-09,	-1,	-7.17959e-09,	1		});
    testJac.emplace_back(std::initializer_list<double>{ 0.951057,	-0.587785,	-0.587785,	0.951057,	1.07694e-08		});
    testJac.emplace_back(std::initializer_list<double>{ 0.809017,	-0.951057,	0.309017,	0.587785,	-1	});
    testJac.emplace_back(std::initializer_list<double>{ 0.587785,	-0.951057,	0.951057,	-0.587785,	-1.43592e-08	});
    testJac.emplace_back(std::initializer_list<double>{ 0.309017,	-0.587785,	0.809017,	-0.951057,	1	});
    testJac.emplace_back(std::initializer_list<double>{ 0,	-0,	0,	-0,	0	});	

    auto jac = basis->getControlJacobian();

    ASSERT_EQ( jac.size() , basis->getN() );
    ASSERT_EQ( jac.front().size() , basis->getM() );

    for(size_t i = 0; i < basis->getN(); i++)
    {
        for(size_t j = 0; j < basis->getM(); j++)
        {
            EXPECT_NEAR(jac[i][j] , testJac[i][j] , 5e-6);
        }
    }
} 


TEST_F(ChoppedSineTest, testConvertHessian)
{
    // test Hessu = zeros
    std::vector< std::vector<double> > Hessu1(basis->getN(), std::vector<double>(basis->getN(), 0));
    auto Hessc1 = basis->convertHessian(Hessu1);

    for(size_t i = 0; i < basis->getM(); i++)
    {
        for(size_t j = 0; j < basis->getM(); j++)
        {
            EXPECT_NEAR(Hessc1[i][j] , 0 , 1e-10);
        }
    }

    // test Hessu = ones
    std::vector< std::vector<double> > Hessu2(basis->getN(), std::vector<double>(basis->getN(), 1));
    auto Hessc2 = basis->convertHessian(Hessu2);

    std::vector< std::vector<double> > testHess2;
    
    testHess2.emplace_back(std::initializer_list<double>{  39.8635  , -0.0000 ,  12.3914  ,  0.0000  ,  6.3138	});
    testHess2.emplace_back(std::initializer_list<double>{  -0.0000  , -0.0000 ,  -0.0000  ,  0.0000  ,  0.0000	});
    testHess2.emplace_back(std::initializer_list<double>{ 12.3914  , -0.0000  ,  3.8518  ,  0.0000   , 1.9626	});
    testHess2.emplace_back(std::initializer_list<double>{ -0.0000  ,  0.0000 ,  -0.0000     ,    0   , 0.0000	});
    testHess2.emplace_back(std::initializer_list<double>{ 6.3138  , -0.0000  ,  1.9626 ,   0.0000  ,  1.0000	});
    

    for(size_t i = 0; i < basis->getM(); i++)
    {
        for(size_t j = 0; j < basis->getM(); j++)
        {
            EXPECT_NEAR(Hessc2[i][j] , testHess2[i][j]  , 1e-4);
        }
    }

    // test Hessu = [1 , 2, 3 ...]
    std::vector< std::vector<double> > Hessu3(basis->getN(), std::vector<double>(basis->getN(), 1));
    double idx = 0.0;
    
    for(size_t i = 0; i < basis->getN(); i++)
    {
        for(size_t j = i; j < basis->getN(); j++)
        {
            Hessu3[i][j] = idx;
            Hessu3[j][i] = idx;
            idx += 0.01;
        }   
    }

    auto Hessc3 = basis->convertHessian(Hessu3);

    std::vector< std::vector<double> > testHess3;
    testHess3.emplace_back(std::initializer_list<double>{  14.8420 ,  -3.5725 ,   3.3413 ,  -1.8170 ,   1.6800	});
    testHess3.emplace_back(std::initializer_list<double>{  -3.5725 ,   1.6547 ,  -0.8321 ,   0.4766 ,  -0.4938  });
    testHess3.emplace_back(std::initializer_list<double>{  3.3413  , -0.8321  ,  1.1382  , -0.3595  ,  0.4339	});
    testHess3.emplace_back(std::initializer_list<double>{  -1.8170 ,   0.4766 ,  -0.3595 ,   0.3759 ,  -0.1662	});
    testHess3.emplace_back(std::initializer_list<double>{  1.6800  , -0.4938  ,  0.4339  , -0.1662  ,  0.3300   });
    

    for(size_t i = 0; i < basis->getM(); i++)
    {
        for(size_t j = 0; j < basis->getM(); j++)
        {
            EXPECT_NEAR(Hessc3[i][j] , testHess3[i][j]  , 1e-4);
        }
    }

} 


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
