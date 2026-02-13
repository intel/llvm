#include <sycl/sycl.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;
namespace syclext = sycl::ext::oneapi;

extern "C" SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (syclexp::nd_range_kernel<1>)) void iota(float start, float *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id] = start + static_cast<float>(id);
}
