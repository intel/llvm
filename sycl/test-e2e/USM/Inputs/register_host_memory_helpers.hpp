// Shared helpers for the sycl_ext_oneapi_register_host_memory end-to-end tests.

#pragma once

#include <cassert>
#include <cstddef>

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
