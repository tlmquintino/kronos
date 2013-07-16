#ifndef kronos_Gemm_h
#define kronos_Gemm_h

#include <string>

#include <boost/filesystem/path.hpp>

#include "MData.h"

namespace kronos {

//------------------------------------------------------------------------------------------

class Gemm {

public: // methods

    Gemm();

    virtual std::string name() = 0;

    /// loads the test data into memory
    void setup(const boost::filesystem::path& p,
               const size_t wn,
               const size_t lat,
               const size_t trc,
               const std::vector<size_t>& fields );

    /// releases the test data from memory
    void run();

    /// verifies the computation by comparing them to reference solutions
    bool verify();

    /// releases the test data from memory
    void teardown();

    /// provides a summary of the test inputs
    std::string prologue();

    /// provides a summary of the test results
    std::string summary();

    void align_to(const size_t& );

    void steps(const size_t& );

    void threads(const size_t& );

    void print_results( bool v ){ print_results_ = v; }

protected: // methods

    /// initializes the computing environment
    virtual void initiate_env() {}

    /// preprocess constant computations ( not dependent on iterations )
    virtual void pre_process() {}

    /// copies in iteration dependent data to device
    virtual void copy_in() {}

    /// performs the actual computations
    virtual void compute() = 0;

    /// copies out iteration dependent data from device
    virtual void copy_out() {}

    /// post-process computations ( not dependent on iterations )
    virtual void post_process() {}

    /// terminates the computing environment
    virtual void terminate_env() {}

protected: // members

    struct Timers
    {
        double copy_in;
        double compute;
        double copy_out;
    };

    Timers timers_;

    real_t norm_L2_;
    double copy_into_;
    double copy_back_;
    double flops_;

    size_t steps_;
    size_t align_to_;
    size_t threads_;
    bool   print_results_;

protected: // members

    MData* md;

    size_t size_A;
    size_t size_B;
    size_t size_C;

    size_t mem_size_A;
    size_t mem_size_B;
    size_t mem_size_C;

    size_t wn_;
    size_t lat_;
    size_t trc_;
    size_t sumf_;

};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
