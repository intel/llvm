// REQUIRES: aspect-ext_oneapi_register_host_memory

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// End-to-end test for sycl_ext_oneapi_register_host_memory. Registers a
// page-aligned host allocation and exercises:
//  - using the registered pointer directly in device code,
//  - explicit USM copies to and from the registered memory,
//  - get_pointer_type reporting usm::alloc::host while registered,
//  - error handling for null pointer and zero size,
//  - registering with the read_only property and reading from it in device
//    code (device writes to a read_only range are undefined behavior and are
//    therefore not exercised).

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/register_host_memory.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <cstdlib>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#else
#include <unistd.h>
#endif

namespace syclexp = sycl::ext::oneapi::experimental;

static size_t getHostPageSize() {
#if defined(_WIN32)
  SYSTEM_INFO Info;
  GetSystemInfo(&Info);
  return static_cast<size_t>(Info.dwPageSize);
#else
  return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
}

static void *allocatePageAligned(size_t Alignment, size_t Size) {
#if defined(_WIN32)
  return _aligned_malloc(Size, Alignment);
#else
  return aligned_alloc(Alignment, Size);
#endif
}

static void freePageAligned(void *Ptr) {
#if defined(_WIN32)
  _aligned_free(Ptr);
#else
  free(Ptr);
#endif
}

int main() {
  sycl::queue Q;
  sycl::context Ctxt = Q.get_context();

  const size_t PageSize = getHostPageSize();
  const size_t NumElems = 1024;
  // Round the byte size up to a multiple of the page size as required.
  size_t NumBytes = NumElems * sizeof(int);
  NumBytes = (NumBytes + PageSize - 1) & ~(PageSize - 1);

  int *Data = static_cast<int *>(allocatePageAligned(PageSize, NumBytes));
  assert(Data != nullptr && "host allocation failed");

  // Error handling: null pointer and zero size must throw errc::invalid.
  {
    bool Threw = false;
    try {
      syclexp::register_host_memory(nullptr, NumBytes, Ctxt);
    } catch (const sycl::exception &E) {
      Threw = (E.code() == sycl::errc::invalid);
    }
    assert(Threw && "null pointer should throw errc::invalid");

    Threw = false;
    try {
      syclexp::register_host_memory(Data, 0, Ctxt);
    } catch (const sycl::exception &E) {
      Threw = (E.code() == sycl::errc::invalid);
    }
    assert(Threw && "zero size should throw errc::invalid");

    // An unaligned pointer must throw errc::invalid.
    Threw = false;
    try {
      syclexp::register_host_memory(reinterpret_cast<char *>(Data) + 64,
                                    NumBytes, Ctxt);
    } catch (const sycl::exception &E) {
      Threw = (E.code() == sycl::errc::invalid);
    }
    assert(Threw && "unaligned pointer should throw errc::invalid");

    // A size that is not a multiple of the page size must throw errc::invalid.
    Threw = false;
    try {
      syclexp::register_host_memory(Data, NumBytes + 1, Ctxt);
    } catch (const sycl::exception &E) {
      Threw = (E.code() == sycl::errc::invalid);
    }
    assert(Threw && "unaligned size should throw errc::invalid");

    // A range whose end address would overflow the host address space must
    // throw errc::invalid.
    Threw = false;
    void *TopPage = reinterpret_cast<void *>(static_cast<uintptr_t>(-1) &
                                             ~(PageSize - 1));
    try {
      syclexp::register_host_memory(TopPage, PageSize, Ctxt);
    } catch (const sycl::exception &E) {
      Threw = (E.code() == sycl::errc::invalid);
    }
    assert(Threw && "non-representable range should throw errc::invalid");
  }

  syclexp::register_host_memory(Data, NumBytes, Ctxt);

  // While registered, the pointer behaves like a USM host allocation.
  assert(sycl::get_pointer_type(Data, Ctxt) == sycl::usm::alloc::host);
  // Interior pointers are also reported as host allocations.
  assert(sycl::get_pointer_type(Data + 1, Ctxt) == sycl::usm::alloc::host);

  // The registered pointer can be referenced directly from device code.
  Q.parallel_for(NumElems, [=](sycl::id<1> I) {
     Data[I] = static_cast<int>(I.get(0)) * 2;
   }).wait();
  for (size_t I = 0; I < NumElems; ++I)
    assert(Data[I] == static_cast<int>(I) * 2);

  // Explicit copies to and from the registered memory.
  std::vector<int> HostSrc(NumElems);
  for (size_t I = 0; I < NumElems; ++I)
    HostSrc[I] = static_cast<int>(I) + 7;

  // Copy from unregistered host memory into the registered range.
  Q.memcpy(Data, HostSrc.data(), NumElems * sizeof(int)).wait();
  for (size_t I = 0; I < NumElems; ++I)
    assert(Data[I] == static_cast<int>(I) + 7);

  // Copy from the registered range back out to plain host memory.
  std::vector<int> HostDst(NumElems, 0);
  Q.memcpy(HostDst.data(), Data, NumElems * sizeof(int)).wait();
  for (size_t I = 0; I < NumElems; ++I)
    assert(HostDst[I] == static_cast<int>(I) + 7);

  syclexp::unregister_host_memory(Data, Ctxt);

  // Register the same range with the read_only property and have device code
  // read (but never write) it, writing results to a separate allocation.
  for (size_t I = 0; I < NumElems; ++I)
    Data[I] = static_cast<int>(I) + 1;

  syclexp::register_host_memory(Data, NumBytes, Ctxt,
                                syclexp::properties{syclexp::read_only});
  assert(sycl::get_pointer_type(Data, Ctxt) == sycl::usm::alloc::host);

  int *Out = sycl::malloc_host<int>(NumElems, Q);
  assert(Out != nullptr && "host allocation failed");
  Q.parallel_for(NumElems, [=](sycl::id<1> I) {
     Out[I] = Data[I] * 2;
   }).wait();
  for (size_t I = 0; I < NumElems; ++I)
    assert(Out[I] == (static_cast<int>(I) + 1) * 2);

  sycl::free(Out, Q);
  syclexp::unregister_host_memory(Data, Ctxt);

  // The application still owns and must free the host memory.
  freePageAligned(Data);

  return 0;
}
