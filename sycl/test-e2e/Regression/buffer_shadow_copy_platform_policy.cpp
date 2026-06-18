// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

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

// Exercises the map/unmap path: a mid-lifetime host_accessor on a live buffer.
// This is the path that, on adapters keeping a separate working buffer (e.g.
// L0v2 integrated with a non-importable pointer), must keep the user pointer
// and the device-visible buffer in sync in BOTH directions. The buffer is
// local to this function, so it is destroyed on return; the caller then checks
// the final write-back at HostPtr.
//
// Sequence: kernel writes I -> host reads (map READ) and verifies -> host adds
// 100 (unmap WRITE-back) -> kernel adds 1 -> scope exit writes I+101 to host.
static bool runMidLifeHostAccessor(sycl::queue &Q, int *HostPtr, size_t N) {
  sycl::buffer<int, 1> Buf(HostPtr, sycl::range<1>(N));

  Q.submit([&](sycl::handler &CGH) {
    auto Acc = Buf.get_access<sycl::access::mode::write>(CGH);
    CGH.single_task([=]() {
      for (size_t I = 0; I < N; ++I)
        Acc[I] = static_cast<int>(I);
    });
  });
  Q.wait_and_throw();

  // Mid-lifetime host access: map READ must observe the kernel's writes, and
  // the modification must survive back to the device on unmap WRITE.
  {
    sycl::host_accessor HAcc(Buf, sycl::read_write);
    for (size_t I = 0; I < N; ++I) {
      if (HAcc[I] != static_cast<int>(I))
        return false;
      HAcc[I] += 100;
    }
  }

  Q.submit([&](sycl::handler &CGH) {
    auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);
    CGH.single_task([=]() {
      for (size_t I = 0; I < N; ++I)
        Acc[I] += 1;
    });
  });
  Q.wait_and_throw();

  return true;
}

// A read-only buffer must never modify the user's source data, regardless of
// whether the backend aliases the pointer (zero-copy) or copies it. Portable
// across all backends because a read-only access performs no writes.
static bool runReadOnlyImmutability(sycl::queue &Q, const int *HostPtr,
                                    size_t N) {
  std::vector<int> Orig(N);
  std::memcpy(Orig.data(), HostPtr, sizeof(int) * N);

  {
    sycl::buffer<int, 1> Buf(HostPtr, sycl::range<1>(N));
    sycl::buffer<int, 1> SumBuf(1);
    Q.submit([&](sycl::handler &CGH) {
      auto In = Buf.get_access<sycl::access::mode::read>(CGH);
      auto S = SumBuf.get_access<sycl::access::mode::write>(CGH);
      CGH.single_task([=]() {
        int Sum = 0;
        for (size_t I = 0; I < N; ++I)
          Sum += In[I];
        S[0] = Sum;
      });
    });
    Q.wait_and_throw();
  }

  return std::memcmp(Orig.data(), HostPtr, sizeof(int) * N) == 0;
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

  // --- Read-only immutability ---
  // A read-only buffer must leave the user's source untouched, aligned or not.
  if (!runReadOnlyImmutability(Q, AlignedInput.data(), N)) {
    std::cerr << "Read-only buffer modified aligned source data\n";
    return 1;
  }
  if (!runReadOnlyImmutability(Q, ReadOnlyUnalignedPtr, N)) {
    std::cerr << "Read-only buffer modified misaligned source data\n";
    return 1;
  }

  // --- Mid-lifetime host_accessor (map/unmap) + final write-back ---
  // Exercises the bidirectional map/unmap sync path and the final copy-back.
  // Expected final pattern at the host pointer: I + 101 (kernel wrote I, host
  // added 100 via map/unmap, kernel added 1).
  auto checkMidLife = [](const int *Ptr) {
    std::vector<int> Tmp(N);
    std::memcpy(Tmp.data(), Ptr, sizeof(int) * N);
    for (size_t I = 0; I < N; ++I)
      if (Tmp[I] != static_cast<int>(I + 101))
        return false;
    return true;
  };

  std::vector<int> AlignedMid(N, 0);
  if (!runMidLifeHostAccessor(Q, AlignedMid.data(), N) ||
      !checkMidLife(AlignedMid.data())) {
    std::cerr << "Mid-life host_accessor failed on aligned buffer\n";
    return 1;
  }

  std::vector<unsigned char> MidStorage(sizeof(int) * N + 1, 0);
  int *UnalignedMidPtr = reinterpret_cast<int *>(MidStorage.data() + 1);
  if (!runMidLifeHostAccessor(Q, UnalignedMidPtr, N) ||
      !checkMidLife(UnalignedMidPtr)) {
    std::cerr << "Mid-life host_accessor failed on misaligned buffer\n";
    return 1;
  }

  return 0;
}
