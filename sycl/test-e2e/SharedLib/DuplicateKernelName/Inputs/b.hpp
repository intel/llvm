#include <sycl/detail/core.hpp>

#include "common.hpp"

class TestKernel;
API_EXPORT void submitKernelDirectB(sycl::queue &Q, int *Ptr);
API_EXPORT void submitKernelWithIdB(sycl::queue &Q, int *Ptr);
