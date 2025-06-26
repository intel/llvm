#include <sycl/sycl.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

typedef void (*FuncPtrT)(size_t *);

struct ArgsT {
  size_t *Ptr;
  FuncPtrT *FuncPtr;
};

SYCL_EXTERNAL size_t GetID() {
  return syclext::this_work_item::get_nd_item<1>().get_global_id();
}

extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (syclexp::nd_range_kernel<1>)) void Kernel(ArgsT Args) {
  (**Args.FuncPtr)(Args.Ptr);
}
