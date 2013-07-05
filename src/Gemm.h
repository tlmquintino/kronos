#ifndef kronos_Gemm_h
#define kronos_Gemm_h

#include <string>

#include <boost/filesystem/path.hpp>

namespace kronos {

//------------------------------------------------------------------------------------------

class Gemm {

public: // methods

    Gemm();

    virtual std::string name() = 0;

    /// loads the test data into memory
    void setup(const boost::filesystem::path& t);

    /// releases the test data from memory
    void run();

    /// verifies the computation by comparing them to reference solutions
    bool verify();

    /// releases the test data from memory
    void teardown();

    /// provides a summary of the test
    std::string summary();

protected: // methods

    /// initializes the computing environment, if necessary
    virtual void initiate() {}

    /// performs the actual computations
    virtual void compute() = 0;

    /// terminates the computing environment, if necessary
    virtual void terminate() {}

protected: // members

    struct {
        double initiate;
        double compute;
        double terminate;
    } timers_ ;

    long long flops_;

    boost::filesystem::path test_;

};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
