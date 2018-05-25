#include "ControlBasis.hpp"

ControlBasis::ControlBasis(stdvec& u0, stdvec& S, rowmat& f)
    : u0(u0), S(S), f(f), N(u0.size()), M(f.front().size())
{
    // build control Jacobian du_i/dc_n
    controlJacobian = f;
    size_t i = 0;
    for (auto& row : controlJacobian)
    {
        for (auto& val : row)
        {
            val *= S[i];
        }
        i++;    
    }

    // set current control to u0 as no c vector specified
    ucurrent = u0;
}

size_t ControlBasis::getM() const
{
    return M;
}

size_t ControlBasis::getN() const
{   
    return N;
}

stdvec ControlBasis::convertControl( const stdvec& control, const bool new_control)
{
    // u = u0 + S*sum_n (f_n * c_n)
    // control = (c_1 , ... , c_M)
    if (new_control) 
    {
        assert(control.size() == M);

        stdvec u = u0;
        for (size_t i = 0; i < N; i++)
        {
            u[i] += S[i]*std::inner_product(f[i].begin(), f[i].end(), control.begin(), 0);
        }

        ucurrent = u;
    }

    return ucurrent;
}


stdvec ControlBasis::convertGradient( const stdvec& gradu ) const
{
    // dJ/dc_n = sum_i (dJ/du_i S_i f_{i,n} ) 
    assert( gradu.size() == N );

    stdvec gradc(M);
    
    for(size_t n = 0; n < M; n++)
    {
        double gn = 0;
        for(size_t i = 0; i < N; i++)
        {
            gn += S[i]*gradu[i]*f[i][n];
        }

        gradc[n] = gn;       
    }
    
    return gradc;
}


rowmat ControlBasis::getControlJacobian() const
{
    return controlJacobian;
}