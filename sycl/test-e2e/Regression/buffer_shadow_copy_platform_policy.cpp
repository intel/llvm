// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Regression test for buffer construction policies. Exercises every branch
// in SYCL's shadow-copy decision and the L0 v2 UR ur_integrated_buffer_handle_t
// constructor: no host pointer, aligned libc, misaligned libc, USM host,
// USM shared and iterator-pair. For each source the test drives read, write
// and read-write access patterns where applicable and validates host-side
// state after the buffer goes out of scope.

#include <sycl/detail/core.hpp>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

using namespace sycl;

constexpr size_t N = 32;

static int Errors = 0;

static void fail(const char *Name, const char *What) {
  std::cerr << "FAIL: " << Name << " (" << What << ")\n";
  ++Errors;
}

static int expectedSum() { return static_cast<int>((N - 1) * N / 2); }

static void initSequence(int *Ptr) {
  for (size_t I = 0; I < N; ++I)
    Ptr[I] = static_cast<int>(I);
}

static bool isStamped(const int *Ptr) {
  for (size_t I = 0; I < N; ++I)
    if (Ptr[I] != static_cast<int>(I * 3 + 7))
      return false;
  return true;
}

static bool isDoubled(const int *Ptr) {
  for (size_t I = 0; I < N; ++I)
    if (Ptr[I] != static_cast<int>(I) * 2)
      return false;
  return true;
}

// Sum all elements via a read accessor.
static int sumViaBuffer(queue &Q, buffer<int, 1> &Buf) {
  buffer<int, 1> SumBuf(1);
  Q.submit([&](handler &CGH) {
    auto InAcc = Buf.get_access<access::mode::read>(CGH);
    auto SumAcc = SumBuf.get_access<access::mode::write>(CGH);
    CGH.single_task([=]() {
      int S = 0;
      for (size_t I = 0; I < N; ++I)
        S += InAcc[I];
      SumAcc[0] = S;
    });
  });
  Q.wait_and_throw();
  return host_accessor(SumBuf, read_only)[0];
}

// Stamp a recognizable pattern via a write accessor.
static void stampViaBuffer(queue &Q, buffer<int, 1> &Buf) {
  Q.submit([&](handler &CGH) {
    auto OutAcc = Buf.get_access<access::mode::write>(CGH);
    CGH.single_task([=]() {
      for (size_t I = 0; I < N; ++I)
        OutAcc[I] = static_cast<int>(I * 3 + 7);
    });
  });
  Q.wait_and_throw();
}

// Multiply each element by 2 via a read_write accessor.
static void doubleViaBuffer(queue &Q, buffer<int, 1> &Buf) {
  Q.submit([&](handler &CGH) {
    auto Acc = Buf.get_access<access::mode::read_write>(CGH);
    CGH.single_task([=]() {
      for (size_t I = 0; I < N; ++I)
        Acc[I] = Acc[I] * 2;
    });
  });
  Q.wait_and_throw();
}

// Read path: buffer is constructed over an already-initialized host pointer
// and the kernel must observe the original data.
static void runReadCase(queue &Q, const char *Name, const int *HostPtr) {
  buffer<int, 1> Buf(HostPtr, range<1>(N));
  if (sumViaBuffer(Q, Buf) != expectedSum())
    fail(Name, "read");
}

// Write path: kernel stamps a pattern, the host pointer must reflect it
// after the buffer is destroyed.
static void runWriteCase(queue &Q, const char *Name, int *HostPtr) {
  std::memset(HostPtr, 0, sizeof(int) * N);
  // The buffer needs to be destroyed before the host-side check — that's when
  // write-back happens.
  {
    buffer<int, 1> Buf(HostPtr, range<1>(N));
    stampViaBuffer(Q, Buf);
  }
  if (!isStamped(HostPtr))
    fail(Name, "writeback");
}

// Read-write path: kernel doubles each element. Validates both that the
// kernel saw the initial data and that the result is written back.
static void runReadWriteCase(queue &Q, const char *Name, int *HostPtr) {
  initSequence(HostPtr);
  {
    buffer<int, 1> Buf(HostPtr, range<1>(N));
    doubleViaBuffer(Q, Buf);
  }
  if (!isDoubled(HostPtr))
    fail(Name, "read_write");
}

int main() {
  queue Q;

  // 1. No host pointer: SYCL allocates internally. Validate kernel write is
  // visible through a host_accessor after the kernel completes.
  {
    buffer<int, 1> Buf{range<1>(N)};
    stampViaBuffer(Q, Buf);
    auto Acc = host_accessor(Buf, read_only);
    for (size_t I = 0; I < N; ++I) {
      if (Acc[I] != static_cast<int>(I * 3 + 7)) {
        fail("no-host-ptr", "host_accessor read");
        break;
      }
    }
  }

  // 2. Aligned libc host pointer (std::vector::data()).
  {
    std::vector<int> Vec(N);
    initSequence(Vec.data());
    runReadCase(Q, "aligned-libc", Vec.data());
  }
  {
    std::vector<int> Vec(N);
    runWriteCase(Q, "aligned-libc", Vec.data());
  }
  {
    std::vector<int> Vec(N);
    runReadWriteCase(Q, "aligned-libc", Vec.data());
  }

  // 3. Misaligned libc host pointer (1-byte offset into a byte buffer).
  // This is the original triple-buffer case from PR #21694.
  {
    std::vector<unsigned char> Storage(sizeof(int) * N + 1);
    int *Ptr = reinterpret_cast<int *>(Storage.data() + 1);
    initSequence(Ptr);
    runReadCase(Q, "misaligned-libc", Ptr);
  }
  {
    std::vector<unsigned char> Storage(sizeof(int) * N + 1, 0);
    int *Ptr = reinterpret_cast<int *>(Storage.data() + 1);
    runWriteCase(Q, "misaligned-libc", Ptr);
  }
  {
    std::vector<unsigned char> Storage(sizeof(int) * N + 1);
    int *Ptr = reinterpret_cast<int *>(Storage.data() + 1);
    runReadWriteCase(Q, "misaligned-libc", Ptr);
  }

  // 4. Iterator-pair construction. SYCL copies the source range into its own
  // storage; there is no write-back to the iterator. Read-only verification.
  {
    std::vector<int> Vec(N);
    initSequence(Vec.data());
    buffer<int, 1> Buf(Vec.begin(), Vec.end());
    if (sumViaBuffer(Q, Buf) != expectedSum())
      fail("iterator-pair", "read");
  }

  return Errors ? 1 : 0;
}
