#include <string_view>

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

template <typename T, typename S>
static int performResultCheck(size_t NumberOfElements, const T *ResultPtr,
                              std::string_view TestName,
                              S ExpectedResultValue) {
  int IsSuccessful{0};
  for (size_t i = 0; i < NumberOfElements; i++) {
    if (ResultPtr[i] != ExpectedResultValue) {
      std::cerr << "Failed " << TestName << " : " << ResultPtr[i]
                << " != " << ExpectedResultValue << std::endl;
      ++IsSuccessful;
    }
  }
  return IsSuccessful;
}

template <auto *Func> static sycl::kernel getKernel(sycl::context &Context) {
  sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundle =
      syclexp::get_kernel_bundle<Func, sycl::bundle_state::executable>(Context);
  sycl::kernel KernelId = KernelBundle.ext_oneapi_get_kernel<Func>();
  return KernelId;
}
