// REQUIRES: aspect-ext_oneapi_register_host_memory
// REQUIRES: level_zero_v2_adapter
// REQUIRES: linux
// REQUIRES: hugepages

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// End-to-end test for sycl_ext_oneapi_register_host_memory with huge-page
// backed host memory. Two huge-page acquisition paths are exercised:
//  - explicit huge pages via mmap(MAP_HUGETLB), and
//  - transparent huge pages via mmap + madvise(MADV_HUGEPAGE).
//
// Explicit huge pages require the OS to have huge pages reserved (e.g. via
// /proc/sys/vm/nr_hugepages). The test therefore REQUIRES the "hugepages"
// feature (free HugeTLB pages present) so it is UNSUPPORTED rather than
// silently skipped where none are available. A huge page is itself a multiple
// of the host base page size, so a huge-page-aligned range trivially satisfies
// the extension's page-alignment and size requirements.

#include "Inputs/register_host_memory_helpers.hpp"

#include <cassert>
#include <fstream>
#include <limits>
#include <string>

#include <sys/mman.h>
#include <unistd.h>

// Older glibc headers may not define MAP_HUGETLB; define it to its well-known
// value so the test still builds. The mmap call will simply fail at runtime if
// the running kernel does not support the flag, and that path is skipped.
#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif

// Returns the kernel's default HugeTLB page size in bytes, read from
// /proc/meminfo. Falls back to 2 MiB (the common x86-64 default) if the entry
// is missing.
static size_t getHugePageSize() {
  std::ifstream Meminfo("/proc/meminfo");
  std::string Key;
  while (Meminfo >> Key) {
    if (Key == "Hugepagesize:") {
      size_t KiB = 0;
      Meminfo >> KiB;
      if (KiB != 0)
        return KiB * 1024;
      break;
    }
    Meminfo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return 2 * 1024 * 1024;
}

// Path 1: explicit huge pages via MAP_HUGETLB. The "hugepages" REQUIRES feature
// guarantees free HugeTLB pages exist, so the mapping is expected to succeed
// when the mapping length matches the kernel's default HugeTLB page size.
static void testExplicitHugePages(sycl::queue &Q, sycl::context &Ctxt) {
  // One whole huge page worth of memory, sized to the kernel's default HugeTLB
  // page size so MAP_HUGETLB accepts the length regardless of platform.
  const size_t NumBytes = getHugePageSize();
  const size_t NumElems = NumBytes / sizeof(int);

  void *Map = mmap(nullptr, NumBytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
  assert(Map != MAP_FAILED && "explicit MAP_HUGETLB mmap failed");
  int *Data = static_cast<int *>(Map);

  registerWriteVerifyUnregister(Q, Ctxt, Data, NumBytes, NumElems);

  assert(munmap(Map, NumBytes) == 0 && "munmap failed");
}

// Path 2: transparent huge pages. madvise(MADV_HUGEPAGE) is a best-effort hint
// and does not change the mapping's address or size, so registration of the
// (base-page-aligned) range is valid regardless of whether the kernel actually
// backs it with a transparent huge page.
static void testTransparentHugePages(sycl::queue &Q, sycl::context &Ctxt) {
  const size_t PageSize = getHostPageSize();
  // Request a region whose length is a multiple of the kernel's HugeTLB page
  // size, giving the kernel a chance to promote it to a transparent huge page.
  // Note: mmap(nullptr, ...) only guarantees base-page alignment for the
  // returned address, not huge-page alignment — this is a best-effort THP hint
  // (size-only), so the promotion is not guaranteed.
  const size_t NumBytes = getHugePageSize();
  const size_t NumElems = NumBytes / sizeof(int);

  void *Map = mmap(nullptr, NumBytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(Map != MAP_FAILED && "anonymous mmap failed");
  assert((reinterpret_cast<uintptr_t>(Map) & (PageSize - 1)) == 0 &&
         "mmap result is not page aligned");

  // Best-effort hint; ignore the result. Even if unsupported, the plain
  // anonymous mapping below is still valid registrable host memory.
  (void)madvise(Map, NumBytes, MADV_HUGEPAGE);

  int *Data = static_cast<int *>(Map);
  registerWriteVerifyUnregister(Q, Ctxt, Data, NumBytes, NumElems);

  assert(munmap(Map, NumBytes) == 0 && "munmap failed");
}

int main() {
  sycl::queue Q;
  sycl::context Ctxt = Q.get_context();

  testExplicitHugePages(Q, Ctxt);
  testTransparentHugePages(Q, Ctxt);

  return 0;
}
