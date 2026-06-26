#include <sycl/detail/core.hpp>

#include "common.hpp"

class TestKernel;
API_EXPORT void submitKernelDirectA(sycl::queue &Q, int *Ptr);
API_EXPORT void submitKernelWithIdA(sycl::queue &Q, int *Ptr);