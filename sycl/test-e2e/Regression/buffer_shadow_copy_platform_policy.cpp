// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Regression test for the triple-buffer issue: when a SYCL buffer is
// constructed with a misaligned or non-USM host pointer, the runtime must
// ensure that:
// 1) (read path)  the kernel observes the original host data correctly;
// 2) (write path) kernel-side modifications are written back to the original
//    host pointer once the buffer goes out of scope.
//
// These two invariants must hold regardless of whether the SYCL runtime or the
// UR adapter is responsible for the internal copy/write-back.  The test is
// intentionally backend-agnostic.

#include <sycl/detail/core.hpp>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

// Read-only kernel: sum all elements and return the result.
static int runReadOnlySumKernel(sycl::queue &Q, const int *HostPtr, size_t N) {
  sycl::buffer<int, 1> Buf(HostPtr, sycl::range<1>(N));
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
    sycl::buffer<int, 1> Buf(HostPtr, sycl::range<1>(N));

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

  // Build aligned reference data.
  std::vector<int> AlignedInput(N);
  for (size_t I = 0; I < N; ++I)
    AlignedInput[I] = static_cast<int>(I);

  // Build a deliberately misaligned copy: offset by 1 byte so that the int*
  // is not naturally aligned.
  std::vector<unsigned char> Storage(sizeof(int) * N + 1);
  int *UnalignedPtr = reinterpret_cast<int *>(Storage.data() + 1);
  std::memcpy(UnalignedPtr, AlignedInput.data(), sizeof(int) * N);
  const int *ReadOnlyUnalignedPtr = UnalignedPtr;

  const int ExpectedSum = static_cast<int>((N - 1) * N / 2);

  // --- Read path correctness ---
  // Both aligned and misaligned host pointers must produce the correct sum.
  const int AlignedSum = runReadOnlySumKernel(Q, AlignedInput.data(), N);
  if (AlignedSum != ExpectedSum) {
    std::cerr << "Unexpected aligned sum: " << AlignedSum << "\n";
    return 1;
  }

  const int MisalignedSum = runReadOnlySumKernel(Q, ReadOnlyUnalignedPtr, N);
  if (MisalignedSum != ExpectedSum) {
    std::cerr << "Unexpected misaligned sum: " << MisalignedSum << "\n";
    return 1;
  }

  // --- Write-back correctness ---
  // After the buffer goes out of scope the kernel-written pattern must be
  // visible at the original host pointer, even when that pointer is misaligned.
  std::vector<int> AlignedWritable(N, 0);
  std::vector<unsigned char> WritableStorage(sizeof(int) * N + 1, 0);
  int *UnalignedWritablePtr =
      reinterpret_cast<int *>(WritableStorage.data() + 1);

  runWriteKernel(Q, AlignedWritable.data(), N);
  runWriteKernel(Q, UnalignedWritablePtr, N);

  if (!checkExpectedPattern(AlignedWritable.data(), N)) {
    std::cerr << "Unexpected data in aligned writable buffer\n";
    return 1;
  }
  if (!checkExpectedPattern(UnalignedWritablePtr, N)) {
    std::cerr << "Unexpected data in misaligned writable buffer\n";
    return 1;
  }

  return 0;
}
