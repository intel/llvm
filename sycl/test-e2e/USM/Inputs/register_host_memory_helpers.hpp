// Shared helpers for the sycl_ext_oneapi_register_host_memory end-to-end tests.

#pragma once

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/register_host_memory.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <cstddef>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

// Returns the host page size. The extension requires registered ranges to be
// aligned to (and a multiple of) the host page size.
inline size_t getHostPageSize() {
#if defined(_WIN32)
  SYSTEM_INFO Info;
  GetSystemInfo(&Info);
  return static_cast<size_t>(Info.dwPageSize);
#else
  return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
}

// Rounds NumBytes up to a whole number of host pages, as the extension requires
// the registered size to be a multiple of the host page size.
inline size_t roundUpToPage(size_t NumBytes, size_t PageSize) {
  assert(PageSize != 0 && (PageSize & (PageSize - 1)) == 0 &&
         "PageSize must be a power of two");
  return (NumBytes + PageSize - 1) & ~(PageSize - 1);
}

// Registers [Data, Data + NumBytes) as writable host memory (NumBytes must be a
// multiple of the host page size), then exercises the registered range and
// unregisters it. The exact device computation does not matter for these tests,
// so a single deterministic kernel is shared: the device writes Base + I into
// the first NumElems ints. The result is verified both directly and through an
// explicit copy-out, and get_pointer_type is checked while registered. The Base
// parameter lets callers distinguish successive registrations of the same range
// (e.g. the re-register test) by writing different values each time.
inline void registerWriteVerifyUnregister(sycl::queue &Q, sycl::context &Ctxt,
                                          int *Data, size_t NumBytes,
                                          size_t NumElems, int Base = 1) {
  namespace syclexp = sycl::ext::oneapi::experimental;
  syclexp::register_host_memory(Data, NumBytes, Ctxt);

  // While registered, the pointer behaves like a USM host allocation.
  assert(sycl::get_pointer_type(Data, Ctxt) == sycl::usm::alloc::host);
  // Interior pointers are reported as host allocations too.
  assert(sycl::get_pointer_type(Data + NumElems / 2, Ctxt) ==
         sycl::usm::alloc::host);

  // Use the registered pointer directly from device code.
  Q.parallel_for(NumElems, [=](sycl::id<1> I) {
     Data[I] = static_cast<int>(I.get(0)) + Base;
   }).wait();
  for (size_t I = 0; I < NumElems; ++I)
    assert(Data[I] == static_cast<int>(I) + Base);

  // An explicit copy out of the registered range yields the same contents.
  std::vector<int> HostDst(NumElems, 0);
  Q.memcpy(HostDst.data(), Data, NumElems * sizeof(int)).wait();
  for (size_t I = 0; I < NumElems; ++I)
    assert(HostDst[I] == static_cast<int>(I) + Base);

  syclexp::unregister_host_memory(Data, Ctxt);
}
