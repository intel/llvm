// SYCLBIN producer that emits an AOT object image with unresolved imported
// symbols. Together with exporting_function.cpp this exercises the load
// path where get_kernel_bundle<bundle_state::object> must surface a native
// (AOT) image whose intrinsic state is object because it still has
// unresolved cross-image dependencies.

#include <sycl/sycl.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

SYCL_EXTERNAL void TestFunc(int *Ptr, int Size);

extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (syclexp::nd_range_kernel<1>)) void AOTKernel(int *Ptr, int Size) {
  size_t I = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  if (static_cast<int>(I) < Size)
    TestFunc(Ptr, Size);
}
