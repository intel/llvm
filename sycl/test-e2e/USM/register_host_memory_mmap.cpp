// REQUIRES: aspect-ext_oneapi_register_host_memory
// REQUIRES: level_zero_v2_adapter
// REQUIRES: linux

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// End-to-end test for sycl_ext_oneapi_register_host_memory with host memory
// obtained from an anonymous private read/write mmap mapping
// (MAP_PRIVATE | MAP_ANONYMOUS). mmap returns page-aligned memory, so the
// registration's page-alignment requirement is satisfied naturally. The
// registered pointer is used directly in device code and read back through an
// explicit copy.

#include "Inputs/register_host_memory_helpers.hpp"

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/register_host_memory.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <vector>

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

  syclexp::register_host_memory(Data, NumBytes, Ctxt);

  // The pointer behaves like a USM host allocation while registered.
  assert(sycl::get_pointer_type(Data, Ctxt) == sycl::usm::alloc::host);

  Q.parallel_for(NumElems, [=](sycl::id<1> I) {
     Data[I] = static_cast<int>(I.get(0)) + 1;
   }).wait();

  std::vector<int> HostDst(NumElems, 0);
  Q.memcpy(HostDst.data(), Data, NumElems * sizeof(int)).wait();
  for (size_t I = 0; I < NumElems; ++I)
    assert(HostDst[I] == static_cast<int>(I) + 1);

  syclexp::unregister_host_memory(Data, Ctxt);
  assert(munmap(Map, NumBytes) == 0 && "munmap failed");

  return 0;
}
