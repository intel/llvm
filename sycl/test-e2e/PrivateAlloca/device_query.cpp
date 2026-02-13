// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the using device query for 'sycl_ext_oneapi_private_alloca' extension
// support, and that the return value matches expectations.

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue Queue;

  sycl::device Device = Queue.get_device();
  bool SupportsPrivateAlloca =
      Device.has(sycl::aspect::ext_oneapi_private_alloca);
  sycl::backend Backend = Device.get_backend();
  bool ShouldSupportPrivateAlloca =
      Backend == sycl::backend::opencl ||
      Backend == sycl::backend::ext_oneapi_level_zero;

  assert(SupportsPrivateAlloca == ShouldSupportPrivateAlloca &&
         "Unexpected support value");
}
