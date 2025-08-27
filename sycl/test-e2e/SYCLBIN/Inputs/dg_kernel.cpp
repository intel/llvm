#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

syclex::device_global<int32_t> DG;

extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (syclex::single_task_kernel)) void ff_dg_adder(int val) {
  DG += val;
}

syclex::device_global<int64_t,
                      decltype(syclex::properties(syclex::device_image_scope))>
    DG_DIS;

extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (syclex::single_task_kernel)) void ff_swap(int64_t *val) {
  int64_t tmp = DG_DIS;
  DG_DIS = *val;
  *val = tmp;
}
