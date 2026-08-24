// REQUIRES: aspect-ext_oneapi_register_host_memory
// REQUIRES: level_zero_v2_adapter
// REQUIRES: linux

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t

// End-to-end test for sycl_ext_oneapi_register_host_memory with host memory
// obtained from a file-backed shared mmap mapping (MAP_SHARED) over a temporary
// file. mmap returns page-aligned memory, so the registration's page-alignment
// requirement is satisfied naturally. The temporary file is created next to
// TmpPrefix (a path within the test's output directory) rather than a global
// location such as /tmp.

#include "Inputs/register_host_memory_helpers.hpp"

#include <cassert>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char **argv) {
  // A path prefix inside the test's output directory is passed as the first
  // argument and is used for the temporary file.
  std::string TmpPrefix = argc > 1 ? argv[1] : "reghostmem";

  sycl::queue Q;
  sycl::context Ctxt = Q.get_context();
  const size_t PageSize = getHostPageSize();

  const size_t NumElems = 1024;
  const size_t NumBytes = roundUpToPage(NumElems * sizeof(int), PageSize);

  std::string Tmpl = TmpPrefix + "_reghostmem_XXXXXX";
  int Fd = mkstemp(Tmpl.data());
  assert(Fd >= 0 && "mkstemp failed");
  // Unlink immediately; the open fd keeps the file alive until close.
  unlink(Tmpl.c_str());
  assert(ftruncate(Fd, static_cast<off_t>(NumBytes)) == 0 &&
         "ftruncate failed");

  void *Map =
      mmap(nullptr, NumBytes, PROT_READ | PROT_WRITE, MAP_SHARED, Fd, 0);
  assert(Map != MAP_FAILED && "file-backed mmap failed");
  int *Data = static_cast<int *>(Map);

  registerWriteVerifyUnregister(Q, Ctxt, Data, NumBytes, NumElems);

  assert(munmap(Map, NumBytes) == 0 && "munmap failed");
  close(Fd);

  return 0;
}
