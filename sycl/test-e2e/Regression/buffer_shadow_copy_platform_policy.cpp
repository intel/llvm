// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Checks that misaligned host-pointer buffers do not allocate a SYCL shadow
// copy on backends where prepareForAllocation() disables it.
//
// The test is portable: expected allocation count is derived from the runtime
// backend, so a single test works across all platforms.

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
    if (MisalignedAllocations != AlignedAllocations) {
      std::cerr << "Unexpected extra allocation on misaligned pointer: aligned="
                << AlignedAllocations
                << ", misaligned=" << MisalignedAllocations << "\n";
      return 1;
    }
  } else {
    if (MisalignedAllocations != AlignedAllocations + 1) {
      std::cerr
          << "Expected one extra allocation for misaligned pointer: aligned="
          << AlignedAllocations << ", misaligned=" << MisalignedAllocations
          << "\n";
      return 1;
    }
  }

  return 0;
}
