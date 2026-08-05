#include <sycl/sycl.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// Kernel reads both spec constants so both are live; the CMPLRLLVM-77316
// collision only triggers when both are referenced.
inline constexpr sycl::specialization_id<int> SC_A{256};
inline constexpr sycl::specialization_id<int> SC_B{1024};

extern "C" SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((
    syclexp::nd_range_kernel<1>)) void spec_const_collision(int *out,
                                                            sycl::kernel_handler
                                                                kh) {
  out[0] = kh.get_specialization_constant<SC_A>();
  out[1] = kh.get_specialization_constant<SC_B>();
}
