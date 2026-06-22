// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

// Read-only kernel: sum all elements and return the result.
static int runReadOnlySumKernel(sycl::queue &Q, const int *HostPtr, size_t N) {
  sycl::buffer<int, 1> Buf(HostPtr, sycl::range<1>(N),
                           {sycl::property::buffer::use_host_ptr{}});
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
    sycl::buffer<int, 1> Buf(HostPtr, sycl::range<1>(N),
                             {sycl::property::buffer::use_host_ptr{}});

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
  sycl::buffer<int, 1> Buf(HostPtr, sycl::range<1>(N),
                           {sycl::property::buffer::use_host_ptr{}});

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
    sycl::buffer<int, 1> Buf(HostPtr, sycl::range<1>(N),
                             {sycl::property::buffer::use_host_ptr{}});
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

  const int ExpectedSum = static_cast<int>((N - 1) * N / 2);

  auto checkMidLife = [](const int *Ptr) {
    std::vector<int> Tmp(N);
    std::memcpy(Tmp.data(), Ptr, sizeof(int) * N);
    for (size_t I = 0; I < N; ++I)
      if (Tmp[I] != static_cast<int>(I + 101))
        return false;
    return true;
  };

  // --- Aligned baseline ---
  if (runReadOnlySumKernel(Q, AlignedInput.data(), N) != ExpectedSum) {
    std::cerr << "Unexpected aligned sum\n";
    return 1;
  }

  std::vector<int> AlignedWritable(N, 0);
  runWriteKernel(Q, AlignedWritable.data(), N);
  if (!checkExpectedPattern(AlignedWritable.data(), N)) {
    std::cerr << "Unexpected data in aligned writable buffer\n";
    return 1;
  }

  if (!runReadOnlyImmutability(Q, AlignedInput.data(), N)) {
    std::cerr << "Read-only buffer modified aligned source data\n";
    return 1;
  }

  std::vector<int> AlignedMid(N, 0);
  if (!runMidLifeHostAccessor(Q, AlignedMid.data(), N) ||
      !checkMidLife(AlignedMid.data())) {
    std::cerr << "Mid-life host_accessor failed on aligned buffer\n";
    return 1;
  }

  // --- Misaligned variants ---
  // Cover several offsets. 1 byte is not even int-aligned; 4 bytes is
  // int-aligned but typically below a backend's required base-address
  // alignment (e.g. CL_DEVICE_MEM_BASE_ADDR_ALIGN is normally 64-128 B);
  // 64 bytes exercises the boundary case where some backends still require
  // a stricter (e.g. 128 B / page) alignment for zero-copy import.
  constexpr size_t Offsets[] = {1, 4, 64};
  for (size_t Offset : Offsets) {
    std::vector<unsigned char> ROStorage(sizeof(int) * N + Offset);
    int *UnalignedPtr = reinterpret_cast<int *>(ROStorage.data() + Offset);
    std::memcpy(UnalignedPtr, AlignedInput.data(), sizeof(int) * N);
    const int *ReadOnlyUnalignedPtr = UnalignedPtr;

    if (runReadOnlySumKernel(Q, ReadOnlyUnalignedPtr, N) != ExpectedSum) {
      std::cerr << "Unexpected misaligned sum (offset=" << Offset << ")\n";
      return 1;
    }

    std::vector<unsigned char> WritableStorage(sizeof(int) * N + Offset, 0);
    int *UnalignedWritablePtr =
        reinterpret_cast<int *>(WritableStorage.data() + Offset);
    runWriteKernel(Q, UnalignedWritablePtr, N);
    if (!checkExpectedPattern(UnalignedWritablePtr, N)) {
      std::cerr << "Unexpected data in misaligned writable buffer (offset="
                << Offset << ")\n";
      return 1;
    }

    if (!runReadOnlyImmutability(Q, ReadOnlyUnalignedPtr, N)) {
      std::cerr << "Read-only buffer modified misaligned source data (offset="
                << Offset << ")\n";
      return 1;
    }

    std::vector<unsigned char> MidStorage(sizeof(int) * N + Offset, 0);
    int *UnalignedMidPtr = reinterpret_cast<int *>(MidStorage.data() + Offset);
    if (!runMidLifeHostAccessor(Q, UnalignedMidPtr, N) ||
        !checkMidLife(UnalignedMidPtr)) {
      std::cerr << "Mid-life host_accessor failed on misaligned buffer (offset="
                << Offset << ")\n";
      return 1;
    }
  }

  // --- Aligned-but-non-importable: raw heap allocation ---
  // A plain new[] returns a pointer that is naturally aligned for int but
  // is not USM-imported, not pinned, and not part of any device-visible
  // allocation. On L0 this exercises the path where maybeImportUSM either
  // succeeds (and the pointer is promoted) or fails and the adapter must
  // fall back to its own backing storage with explicit copies.
  {
    std::unique_ptr<int[]> HeapInput(new int[N]);
    for (size_t I = 0; I < N; ++I)
      HeapInput[I] = static_cast<int>(I);

    if (runReadOnlySumKernel(Q, HeapInput.get(), N) != ExpectedSum) {
      std::cerr << "Unexpected sum on aligned heap pointer\n";
      return 1;
    }

    std::unique_ptr<int[]> HeapWritable(new int[N]());
    runWriteKernel(Q, HeapWritable.get(), N);
    if (!checkExpectedPattern(HeapWritable.get(), N)) {
      std::cerr << "Unexpected data in aligned heap writable buffer\n";
      return 1;
    }

    if (!runReadOnlyImmutability(Q, HeapInput.get(), N)) {
      std::cerr << "Read-only buffer modified aligned heap source data\n";
      return 1;
    }

    std::unique_ptr<int[]> HeapMid(new int[N]());
    if (!runMidLifeHostAccessor(Q, HeapMid.get(), N) ||
        !checkMidLife(HeapMid.get())) {
      std::cerr << "Mid-life host_accessor failed on aligned heap buffer\n";
      return 1;
    }
  }

  return 0;
}
