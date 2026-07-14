// REQUIRES: aspect-ext_oneapi_register_host_memory
// REQUIRES: level_zero_v2_adapter
// REQUIRES: linux

// UNSUPPORTED: aspect-ext_oneapi_is_integrated_gpu
// UNSUPPORTED-TRACKER: GSD-13000

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// End-to-end test for sycl_ext_oneapi_register_host_memory with a read-only
// mmap'd host mapping. The host writes the contents while the mapping is still
// writable, then drops write permission via mprotect before registering,
// mirroring how an application might register a read-only mapping. Device code
// only reads it.

#include "Inputs/register_host_memory_helpers.hpp"

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/register_host_memory.hpp>
#include <sycl/usm.hpp>

#include <cassert>

#include <sys/mman.h>
#include <unistd.h>

namespace syclexp = sycl::ext::oneapi::experimental;

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

  for (size_t I = 0; I < NumElems; ++I)
    Data[I] = static_cast<int>(I) * 3;

  // Make the mapping read-only on the host. The read_only property allows
  // registering host memory that the application cannot write.
  assert(mprotect(Map, NumBytes, PROT_READ) == 0 && "mprotect failed");

  syclexp::register_host_memory(Data, NumBytes, Ctxt,
                                syclexp::properties{syclexp::read_only});
  assert(sycl::get_pointer_type(Data, Ctxt) == sycl::usm::alloc::host);

  // Device code reads the read_only range, writing results elsewhere.
  int *Out = sycl::malloc_shared<int>(NumElems, Q);
  assert(Out != nullptr && "shared allocation failed");
  Q.parallel_for(NumElems, [=](sycl::id<1> I) { Out[I] = Data[I] + 1; }).wait();
  for (size_t I = 0; I < NumElems; ++I)
    assert(Out[I] == static_cast<int>(I) * 3 + 1);

  sycl::free(Out, Q);
  syclexp::unregister_host_memory(Data, Ctxt);
  assert(munmap(Map, NumBytes) == 0 && "munmap failed");

  return 0;
}
