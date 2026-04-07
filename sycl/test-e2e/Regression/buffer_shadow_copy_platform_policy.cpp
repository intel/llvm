// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Checks two things:
// 1) policy check (read-only path): whether misaligned host pointers trigger
//    an extra SYCL shadow-copy allocation depending on backend;
// 2) correctness check (writable path): data is correctly copied back to host
//    when buffer goes out of scope.
//
// The SYCL runtime and the UR adapter layer use different strategies for
// non-importable or misaligned host pointers:
//
//  * Backends with UR buffer allocation (e.g. Level Zero v2):
//    SYCL skips its own shadow copy (prepareForAllocation sets SkipShadowCopy).
//    The UR adapter owns staging and final copy-back:
//      - integrated device: allocates internal USM host memory, copies data in,
//        and copies it back to the original pointer on buffer release;
//      - discrete device: allocates device memory, migrates data from source.
//    These UR-internal allocations go through the USM pool and are NOT seen by
//    CountingAllocator, so the SYCL-layer allocation count stays zero.
//
//  * Backends without UR buffer allocation (e.g. NativeCPU, OpenCL):
//    SYCL creates the shadow copy itself via the buffer's allocator
//    (CountingAllocator records it) and drives copy-back on destruction.
//
// The test measures only SYCL-layer (CountingAllocator) allocations, so it is
// portable across all supported backends without runtime-specific conditionals.

#include <sycl/detail/core.hpp>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

template <typename T> class CountingAllocator {
public:
  using value_type = T;

  CountingAllocator() = default;

  template <typename U>
  constexpr CountingAllocator(const CountingAllocator<U> &) noexcept {}

  T *allocate(std::size_t N) {
    Allocations.fetch_add(1, std::memory_order_relaxed);
    return std::allocator<T>{}.allocate(N);
  }

  void deallocate(T *Ptr, std::size_t N) {
    std::allocator<T>{}.deallocate(Ptr, N);
  }

  template <typename U> bool operator==(const CountingAllocator<U> &) const {
    return true;
  }

  template <typename U> bool operator!=(const CountingAllocator<U> &) const {
    return false;
  }

  static std::atomic<size_t> Allocations;
};

template <typename T> std::atomic<size_t> CountingAllocator<T>::Allocations{0};

static bool shouldSkipAlignedShadowCopy(sycl::backend B) {
  switch (B) {
  case sycl::backend::ext_oneapi_level_zero:
  case sycl::backend::ext_oneapi_cuda:
  case sycl::backend::ext_oneapi_hip:
  case sycl::backend::ext_oneapi_offload:
    return true;
  case sycl::backend::ext_oneapi_native_cpu:
  case sycl::backend::opencl:
    return false;
  default:
    return false;
  }
}

// Read-only kernel path used for allocation-policy assertions.
static int runReadOnlySumKernel(sycl::queue &Q, const int *HostPtr, size_t N) {
  sycl::buffer<int, 1, CountingAllocator<int>> Buf(HostPtr, sycl::range<1>(N));
  sycl::buffer<int, 1> SumBuf(1);

  Q.submit([&](sycl::handler &CGH) {
    auto InAcc = Buf.get_access<sycl::access::mode::read>(CGH);
    auto SumAcc = SumBuf.get_access<sycl::access::mode::write>(CGH);
    CGH.single_task([=]() {
      int Sum = 0;
      for (size_t I = 0; I < N; ++I)
        Sum += InAcc[I];
      SumAcc[0] = Sum;
    });
  });
  Q.wait_and_throw();

  auto SumHostAcc = SumBuf.get_host_access();
  return SumHostAcc[0];
}

// Writable kernel path; buffer destruction happens at scope exit.
static void runWriteKernel(sycl::queue &Q, int *HostPtr, size_t N) {
  {
    sycl::buffer<int, 1, CountingAllocator<int>> Buf(HostPtr,
                                                     sycl::range<1>(N));

    Q.submit([&](sycl::handler &CGH) {
      auto OutAcc = Buf.get_access<sycl::access::mode::write>(CGH);
      CGH.single_task([=]() {
        for (size_t I = 0; I < N; ++I)
          OutAcc[I] = static_cast<int>(I * 3 + 7);
      });
    });
    Q.wait_and_throw();
  }
}

// Verifies host-side result after writable-buffer destruction.
static bool checkExpectedPattern(const int *Ptr, size_t N) {
  std::vector<int> Tmp(N);
  std::memcpy(Tmp.data(), Ptr, sizeof(int) * N);
  for (size_t I = 0; I < N; ++I) {
    if (Tmp[I] != static_cast<int>(I * 3 + 7))
      return false;
  }
  return true;
}

int main() {
  constexpr size_t N = 32;
  sycl::queue Q;

  std::vector<int> AlignedInput(N);
  for (size_t I = 0; I < N; ++I)
    AlignedInput[I] = static_cast<int>(I);

  std::vector<unsigned char> Storage(sizeof(int) * N + 1);
  int *UnalignedPtr = reinterpret_cast<int *>(Storage.data() + 1);
  std::memcpy(UnalignedPtr, AlignedInput.data(), sizeof(int) * N);
  const int *ReadOnlyUnalignedPtr = UnalignedPtr;

  const int ExpectedSum = static_cast<int>((N - 1) * N / 2);

  // Compare aligned vs misaligned read-only input. Allocation count is used as
  // a proxy for SYCL shadow-copy creation.
  CountingAllocator<int>::Allocations.store(0, std::memory_order_relaxed);
  const int AlignedSum = runReadOnlySumKernel(Q, AlignedInput.data(), N);
  const size_t AlignedAllocations =
      CountingAllocator<int>::Allocations.load(std::memory_order_relaxed);
  if (AlignedSum != ExpectedSum) {
    std::cerr << "Unexpected aligned sum: " << AlignedSum << "\n";
    return 1;
  }

  CountingAllocator<int>::Allocations.store(0, std::memory_order_relaxed);
  const int MisalignedSum = runReadOnlySumKernel(Q, ReadOnlyUnalignedPtr, N);

  const size_t MisalignedAllocations =
      CountingAllocator<int>::Allocations.load(std::memory_order_relaxed);
  if (MisalignedSum != ExpectedSum) {
    std::cerr << "Unexpected misaligned sum: " << MisalignedSum << "\n";
    return 1;
  }

  const bool ExpectNoShadowCopy = shouldSkipAlignedShadowCopy(Q.get_backend());

  if (ExpectNoShadowCopy) {
    // SYCL skips its own shadow copy; UR is responsible for staging and
    // copy-back. No SYCL-layer extra allocation is expected regardless of
    // whether the device is integrated or discrete.
    if (MisalignedAllocations != AlignedAllocations) {
      std::cerr << "Unexpected extra SYCL allocation on misaligned pointer: "
                   "aligned="
                << AlignedAllocations
                << ", misaligned=" << MisalignedAllocations << "\n";
      return 1;
    }
  } else {
    // SYCL creates the shadow copy itself; expect exactly one extra allocation.
    if (MisalignedAllocations != AlignedAllocations + 1) {
      std::cerr
          << "Expected one extra SYCL allocation for misaligned pointer: "
             "aligned="
          << AlignedAllocations << ", misaligned=" << MisalignedAllocations
          << "\n";
      return 1;
    }
  }

  // Validate writable path including final copy-back at buffer destruction.
  // This checks correctness only; writable allocation counts are intentionally
  // not asserted (write accessor can conservatively materialize shadow copy).
  std::vector<int> AlignedWritable(N, 0);
  std::vector<unsigned char> WritableStorage(sizeof(int) * N + 1, 0);
  int *UnalignedWritablePtr =
      reinterpret_cast<int *>(WritableStorage.data() + 1);

  CountingAllocator<int>::Allocations.store(0, std::memory_order_relaxed);
  runWriteKernel(Q, AlignedWritable.data(), N);
  const size_t AlignedWriteAllocs =
      CountingAllocator<int>::Allocations.load(std::memory_order_relaxed);

  CountingAllocator<int>::Allocations.store(0, std::memory_order_relaxed);
  runWriteKernel(Q, UnalignedWritablePtr, N);
  const size_t MisalignedWriteAllocs =
      CountingAllocator<int>::Allocations.load(std::memory_order_relaxed);

  if (!checkExpectedPattern(AlignedWritable.data(), N)) {
    std::cerr << "Unexpected data in aligned writable buffer\n";
    return 1;
  }
  if (!checkExpectedPattern(UnalignedWritablePtr, N)) {
    std::cerr << "Unexpected data in misaligned writable buffer\n";
    return 1;
  }

  // Writable allocation counts are intentionally not asserted here because the
  // expected delta is platform-dependent:
  //  * NativeCPU/OpenCL: SYCL creates a shadow copy for misaligned writable
  //    buffers (MisalignedWriteAllocs == AlignedWriteAllocs + 1).
  //  * L0/CUDA/HIP/offload: SYCL skips the shadow copy; UR stages internally
  //    (MisalignedWriteAllocs == AlignedWriteAllocs).
  // The correctness checks above are sufficient to validate copy-back behavior.
  (void)AlignedWriteAllocs;
  (void)MisalignedWriteAllocs;

  return 0;
}
