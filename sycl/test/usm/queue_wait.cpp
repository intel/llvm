// XFAIL: cuda
// piextUSM*Alloc functions for CUDA are not behaving as described in
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/USM.adoc
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/cl_intel_unified_shared_memory.asciidoc
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <cassert>
#include <cstddef>

using namespace cl::sycl;

// This test checks that queue USM functions are properly waited for during
// calls to queue::wait().

int main() {
  const std::size_t Size = 32;
  queue Q;
  std::cout << Q.is_host() << std::endl;
  device Dev = Q.get_device();
  context Ctx = Q.get_context();
  if (!(Dev.get_info<info::device::usm_device_allocations>() &&
        Dev.get_info<info::device::usm_host_allocations>()))
    return 0;

  unsigned char *DevArr = (unsigned char *)malloc_device(Size, Dev, Ctx);
  assert(DevArr);
  unsigned char *HostArr = (unsigned char *)malloc_host(Size, Ctx);
  assert(HostArr);

  Q.memset(DevArr, 42, Size);
  Q.wait();
  Q.memcpy(HostArr, DevArr, Size);
  Q.wait();

  for (std::size_t i = 0; i < Size; ++i)
    assert(HostArr[i] == 42);

  free(DevArr, Ctx);
  free(HostArr, Ctx);

  return 0;
}
