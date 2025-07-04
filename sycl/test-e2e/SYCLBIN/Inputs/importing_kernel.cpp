
#include <sycl/sycl.hpp>

SYCL_EXTERNAL void TestFunc(int *Ptr, int Size);

// use extern "C" to avoid name mangling
extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::
         single_task_kernel)) void TestKernel1(int *Ptr, int Size) {
  TestFunc(Ptr, Size);
}
