// REQUIRES: aspect-ext_oneapi_register_host_memory
// REQUIRES: level_zero_v2_adapter
// REQUIRES: linux
// REQUIRES: hugepages

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s

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

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/register_host_memory.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

// Older glibc headers may not define MAP_HUGETLB; define it to its well-known
// value so the test still builds. The mmap call will simply fail at runtime if
// the running kernel does not support the flag, and that path is skipped.
#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif

namespace syclexp = sycl::ext::oneapi::experimental;

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

// Runs device + copy exercises over a registered range and verifies results.
// Reused by both huge-page paths.
static void exerciseRegisteredRange(sycl::queue &Q, sycl::context &Ctxt,
                                    int *Data, size_t NumElems) {
  syclexp::register_host_memory(Data, NumElems * sizeof(int), Ctxt);

  // The pointer behaves like a USM host allocation while registered.
  assert(sycl::get_pointer_type(Data, Ctxt) == sycl::usm::alloc::host);
  // Interior pointers are reported as host allocations too.
  assert(sycl::get_pointer_type(Data + NumElems / 2, Ctxt) ==
         sycl::usm::alloc::host);

  // Use the registered pointer directly from device code.
  Q.parallel_for(NumElems, [=](sycl::id<1> I) {
     Data[I] = static_cast<int>(I.get(0)) + 11;
   }).wait();
  for (size_t I = 0; I < NumElems; ++I)
    assert(Data[I] == static_cast<int>(I) + 11);

  // Explicit copy out of the registered range.
  std::vector<int> HostDst(NumElems, 0);
  Q.memcpy(HostDst.data(), Data, NumElems * sizeof(int)).wait();
  for (size_t I = 0; I < NumElems; ++I)
    assert(HostDst[I] == static_cast<int>(I) + 11);

  syclexp::unregister_host_memory(Data, Ctxt);
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

  exerciseRegisteredRange(Q, Ctxt, Data, NumElems);

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
  exerciseRegisteredRange(Q, Ctxt, Data, NumElems);

  assert(munmap(Map, NumBytes) == 0 && "munmap failed");
}

int main() {
  sycl::queue Q;
  sycl::context Ctxt = Q.get_context();

  testExplicitHugePages(Q, Ctxt);
  testTransparentHugePages(Q, Ctxt);

  // CHECK: Done (explicit and transparent huge pages tested).
  printf("Done (explicit and transparent huge pages tested).\n");
  return 0;
}
