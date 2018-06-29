#include "ControlBasis.hpp"

ControlBasis::ControlBasis()
{
}


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

    // build matrix of S*f vectors (= transpose of control Jacobian)
    vmat = rowmat(M, std::vector<double>(N, 0));
    
    for(size_t i = 0; i < M; i++)
    {       
        for(size_t j = 0; j < N; j++)
        {
            vmat[i][j] = controlJacobian[j][i];
        }
    }
        
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
            u[i] += S[i]*std::inner_product(f[i].begin(), f[i].end(), control.begin(), 0.0);
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


rowmat ControlBasis::convertHessian( const rowmat& Hessu ) const
{
    assert( Hessu.size() == N );
    assert( Hessu.front().size() == N );

    rowmat Hessc(M, std::vector<double>(M, 0));
    
    for(size_t i = 0; i < M; i++)
    {
        stdvec vi = vmat[i];
        for(size_t j = i; j < M; j++)
        {
            stdvec vj = vmat[j];
            stdvec Hvj;
            Hvj.reserve(N);
            
            for(size_t k = 0; k < N; k++)
            {
                Hvj.push_back( std::inner_product(Hessu[k].begin(), Hessu[k].end(), vj.begin(), 0.0) );
            }

            Hessc[i][j] = std::inner_product(vi.begin(), vi.end(), Hvj.begin(), 0.0);
            Hessc[j][i] = Hessc[i][j];
        }       
    }
    
    return Hessc;
}


rowmat ControlBasis::getControlJacobian() const
{
    return controlJacobian;
}