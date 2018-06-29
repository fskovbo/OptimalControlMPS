#ifndef BH_SITES_H
#define BH_SITES_H
#include "itensor/mps/siteset.h"

namespace itensor {

class BosonSite;

template<typename SiteType>
class BosonSiteSet : public SiteSet
    {
    public:

    BosonSiteSet() { }

    // BosonSiteSet diffrent from standard ITensor SiteSet,
    // as local Fock space can be set on initialization 
    BosonSiteSet(int N, int d)
    {
        auto sites = SiteStore(N);
        for(int j = 1; j <= N; ++j){
            sites.set(j,SiteType(j,d));
        }
        SiteSet::init(std::move(sites));
    }

    BosonSiteSet(std::vector<IQIndex> const& inds)
    {
        int N = inds.size();
        auto sites = SiteStore(N);
        for(int j = 1, i = 0; j <= N; ++j, ++i) {
            auto& Ii = inds.at(i);
            sites.set(j,SiteType(Ii));
        }
        SiteSet::init(std::move(sites));
    }

    // ITensor read is overloaded to reflect non-standard SiteSet constructor
    void read(std::istream& s)
    {
      int N = itensor::read<int>(s);

      if(N > 0) {
        auto store = SiteStore(N);
        for(int j = 1; j <= N; ++j) {
          auto I = IQIndex{};
          I.read(s);
          int d = I.nblock()-1; // infer Fock space dimension d from length/size of index
          store.set(j,SiteType(I,d));
        }
        SiteSet::init(std::move(store));
      }
    }

  };

// BosonSiteSet is initialized using constructor name BoseHubbard()
using BoseHubbard = BosonSiteSet<BosonSite>;

class BosonSite {
    IQIndex s;
    int d;
    std::vector<std::string> stateNames;

    public:

    BosonSite() { }

    BosonSite(IQIndex I) : s(I) { }

    BosonSite(IQIndex I, int d) : s(I) , d(d) { }

    BosonSite(int n, int d)
    : d(d)
    {
        char space[1] = { ' ' };
        auto v = stdx::reserve_vector<IndexQN>(1+d);
        v.emplace_back(Index(nameint("Emp ",n),1,Site),QN("Nb=",0));
        stateNames.emplace_back("Emp");

        // Build possible states of BosonSite:
        // "Emp", "Occ1", "Occ2", ... , "Occd"
        for (size_t i = 1; i <= d; i++) {
            auto name = nameint("Occ",i);
            stateNames.emplace_back(name);
            name += space;
            v.emplace_back(Index(nameint(name,n),1,Site),QN("Nb=",i));
        }

        s = IQIndex(nameint("Boson ",n), std::move(v) );
    }

    IQIndex index() const { return s; }

    // returns index corresponding to requested state
    IQIndexVal state(std::string const& state)
    {
        for (size_t j = 0; j < stateNames.size(); j++) {
            if (state == stateNames.at(j) || state == nameint("",j)) {return s(j+1);}
        }

        Error("State " + state + " not recognized");
        return IQIndexVal{};
    }

    // Returns tensor operator acting solely on this site. 
    // Current operators are:
    // "N"      : number operator 
    // "A"      : annihilation operator
    // "Adag"   : creation operator
    // "N(N-1)" : pair-counting operator (needed as ITensor sometimes has issues
    //                                    with multiple operators on the same site)
    // "Id"     : identity operator
	IQTensor op(std::string const& opname, Args const& args) const
    {
        // Builds base operator tensor with appropriate indices
        auto sP = prime(s);

        std::vector<IQIndexVal> indices(d+1);
        std::vector<IQIndexVal> indicesP(d+1);
        for (size_t j = 0; j <= d; j++) {
            indices.at(j) = s(j+1);
            indicesP.at(j) = sP(j+1);
        }

        IQTensor Op(dag(s),sP);

        // Sets index values according to requested operator
        if(opname == "N")
        {
            for (size_t j = 0; j <= d; j++) {
                Op.set(indices.at(j),indicesP.at(j),j);
            }
        }
        else
        if(opname == "A")
        {
            for (size_t j = 1; j <= d; j++) {
                Op.set(indices.at(j),indicesP.at(j-1),std::sqrt(j));
            }
        }
        else
        if(opname == "Adag")
        {
            for (size_t j = 1; j <= d; j++) {
                Op.set(indices.at(j-1),indicesP.at(j),std::sqrt(j));
            }
        }
        else
	    if(opname == "N(N-1)")
	    {
            for (size_t j = 1; j<= d; j++) {
                Op.set(indices.at(j),indicesP.at(j),j*j-j);
            }
	    }
        else
        if(opname == "NN")
	    {
            for (size_t j = 1; j<= d; j++) {
                Op.set(indices.at(j),indicesP.at(j),j*j);
            }
	    }
        else
        if(opname == "Id")
        {
            for (size_t j = 1; j<= d; j++) {
                Op.set(indices.at(j),indicesP.at(j),1);
            }
        }
	    else
        {
            Error("Operator \"" + opname + "\" name not recognized");
        }

        return Op;
        }
    };


} //namespace itensor

#endif
