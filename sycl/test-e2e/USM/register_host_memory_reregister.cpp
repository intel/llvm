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

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/register_host_memory.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace syclexp = sycl::ext::oneapi::experimental;

// Registers Data as host memory, has the device write Base + I into element I,
// verifies the result, then unregisters. All registrations are writable.
static void registerWriteVerifyUnregister(sycl::queue &Q, sycl::context &Ctxt,
                                          int *Data, size_t NumElems,
                                          int Base) {
  syclexp::register_host_memory(Data, NumElems * sizeof(int), Ctxt);

  // While registered, the pointer behaves like a USM host allocation.
  assert(sycl::get_pointer_type(Data, Ctxt) == sycl::usm::alloc::host);

  Q.parallel_for(NumElems, [=](sycl::id<1> I) {
     Data[I] = static_cast<int>(I.get(0)) + Base;
   }).wait();
  for (size_t I = 0; I < NumElems; ++I)
    assert(Data[I] == static_cast<int>(I) + Base);

  syclexp::unregister_host_memory(Data, Ctxt);
}

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
  registerWriteVerifyUnregister(Q, Ctxt, Data, NumElems, /*Base=*/1);

  // Second registration cycle reusing the same virtual address. The device
  // write here must land in the range after the first cycle was unregistered.
  registerWriteVerifyUnregister(Q, Ctxt, Data, NumElems, /*Base=*/100);

  assert(munmap(Map, NumBytes) == 0 && "munmap failed");

  return 0;
}
