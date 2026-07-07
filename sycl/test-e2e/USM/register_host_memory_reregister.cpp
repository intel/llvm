// REQUIRES: aspect-ext_oneapi_register_host_memory
// REQUIRES: level_zero_v2_adapter
// REQUIRES: linux

// UNSUPPORTED: true
// UNSUPPORTED-TRACKER: GSD-12994

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// End-to-end test for sycl_ext_oneapi_register_host_memory covering repeated
// registration of host memory in a single process. A page-aligned anonymous
// read/write mmap mapping is registered, used from device code, and
// unregistered; the same host range is then re-registered (writable) and
// written to again by the device. This exercises that unregistering a range
// fully releases it so a later registration of the same virtual address starts
// from clean device state.

#include "Inputs/register_host_memory_helpers.hpp"

#include <cassert>

#include <sys/mman.h>
#include <unistd.h>

int main() {
  sycl::queue Q;
  sycl::context Ctxt = Q.get_context();
  const size_t PageSize = getHostPageSize();

  const size_t NumElems = 1024;
  const size_t NumBytes = roundUpToPage(NumElems * sizeof(int), PageSize);

  void *Map = mmap(nullptr, NumBytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(Map != MAP_FAILED && "anonymous mmap failed");
  int *Data = static_cast<int *>(Map);

  // First registration cycle over the range.
  registerWriteVerifyUnregister(Q, Ctxt, Data, NumBytes, NumElems, /*Base=*/1);

  // Second registration cycle reusing the same virtual address. The device
  // write here must land in the range after the first cycle was unregistered.
  registerWriteVerifyUnregister(Q, Ctxt, Data, NumBytes, NumElems,
                                /*Base=*/100);

  assert(munmap(Map, NumBytes) == 0 && "munmap failed");

  return 0;
}
