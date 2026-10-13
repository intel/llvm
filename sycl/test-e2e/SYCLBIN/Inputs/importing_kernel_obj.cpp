// SYCLBIN side of the cross-origin link tests: defines a kernel that imports
// a SYCL_EXTERNAL function which is provided by the RTC-compiled source
// bundle. Exercises the case symmetric to exporting_function.cpp +
// importing kernel from RTC: here the kernel itself lives in the SYCLBIN
// object and must remain reachable after sycl::link({RTC_obj, SYCLBIN_obj}).

#include <sycl/ext/oneapi/free_function_kernel_properties.hpp>
#include <sycl/khr/work_item_queries.hpp>

namespace syclkhr = sycl::khr;
namespace syclexp = sycl::ext::oneapi::experimental;

SYCL_EXTERNAL void TestFunc(int *Ptr, int Size);

extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (syclexp::nd_range_kernel<1>)) void TestKernelBin(int *Ptr, int Size) {
  size_t I = syclkhr::this_nd_item<1>().get_global_linear_id();
  if (static_cast<int>(I) < Size)
    TestFunc(Ptr, Size);
}
