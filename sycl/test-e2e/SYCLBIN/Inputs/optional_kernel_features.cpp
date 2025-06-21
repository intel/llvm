#include <sycl/sycl.hpp>

extern "C" SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::
         single_task_kernel)) void TestKernel1(int *Ptr, int Size) {
  for (size_t I = 0; I < Size; ++I)
    Ptr[I] = I;
}

extern "C" SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::
         single_task_kernel)) void TestKernel2(int *Ptr, int Size) {
  for (size_t I = 0; I < Size; ++I)
    Ptr[I] = static_cast<int>(static_cast<double>(I) / 2.0);
}
