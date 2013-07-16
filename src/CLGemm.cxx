#include "CLGemm.h"

kronos::CLGemm::CLGemm()
{
}

void kronos::CLGemm::initiate_env()
{
    /// @todo implement env initiate
}

void kronos::CLGemm::copy_in()
{
    /// @todo implement memory copy

    real_t* B = &md->B.data()[0];

    copy_into_ += mem_size_B;
}

void kronos::CLGemm::compute()
{
    /// @todo implement compute gemm
}

void kronos::CLGemm::copy_out()
{
    real_t* C = &md->C.data()[0];

    /// @todo implement memory copy

    copy_back_ += mem_size_C;
}

void kronos::CLGemm::terminate_env()
{
    /// @todo implement env terminate
}
