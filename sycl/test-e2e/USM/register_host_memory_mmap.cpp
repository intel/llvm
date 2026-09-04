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

  registerWriteVerifyUnregister(Q, Ctxt, Data, NumBytes, NumElems);

  assert(munmap(Map, NumBytes) == 0 && "munmap failed");

  return 0;
}
