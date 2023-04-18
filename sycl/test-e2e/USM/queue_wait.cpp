// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <iostream>
#include <sycl/sycl.hpp>

#include <cassert>
#include <cstddef>

using namespace sycl;

// This test checks that queue USM functions are properly waited for during
// calls to queue::wait().

int main() {
  const std::size_t Size = 32;
  queue Q;
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

  try {
    Q.memset(nullptr, 42, Size);
    Q.wait_and_throw();
    assert(false && "Expected to have an exception throw instead of assert");
  } catch (runtime_error e) {
  }
  try {
    Q.memcpy(nullptr, DevArr, Size);
    Q.wait_and_throw();
    assert(false && "Expected to have an exception throw instead of assert");
  } catch (runtime_error e) {
  }

  Q.memset(nullptr, 42, 0);
  Q.wait();
  Q.memcpy(nullptr, DevArr, 0);
  Q.wait();

  for (std::size_t i = 0; i < Size; ++i)
    assert(HostArr[i] == 42);

  free(DevArr, Ctx);
  free(HostArr, Ctx);

  std::cout << "Passed\n";
  return 0;
}
