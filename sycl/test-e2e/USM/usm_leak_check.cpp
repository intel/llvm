// REQUIRES: level_zero

// https://github.com/intel/llvm/issues/11434
// XFAIL: gpu-intel-dg2

// RUN: %{build} -o %t.out

// RUN: env UR_L0_LEAKS_DEBUG=1 %{run} %t.out u 2>&1 | FileCheck %s --check-prefix CHECK-USM
// RUN: env UR_L0_LEAKS_DEBUG=1 %{run} %t.out s 2>&1 | FileCheck %s --check-prefix CHECK-SMALL-BUF
// RUN: env UR_L0_LEAKS_DEBUG=1 %{run} %t.out l 2>&1 | FileCheck %s --check-prefix CHECK-LARGE-BUF

#include <sycl/sycl.hpp>
using namespace sycl;

#include <array>
#include <iostream>

void direct_usm(queue &Q) {
  auto p1 = malloc_shared(1024, Q);
  auto p2 = malloc_host(1024, Q);
  auto p3 = malloc_device(1024, Q);
  // Host and Device allocations, pooled by default will be automatically freed
  // Shared is not pooled by default, so it needs to be explicitly freed
  free(p1, Q.get_context());
}

template <typename T> class K;

template <typename T, size_t N> void sycl_buffer(queue &Q) {
  std::array<T, N> A, B, C;
  range<1> numElems{N};
  buffer<T, 1> bufferA(A.data(), numElems);
  buffer<T, 1> bufferB(B.data(), numElems);
  buffer<T, 1> bufferC(C.data(), numElems);

  Q.submit([&](handler &cgh) {
    accessor accA{bufferA, cgh, read_only};
    accessor accB{bufferB, cgh, read_only};
    accessor accC{bufferC, cgh, write_only};

    cgh.parallel_for<class K<T>>(numElems,
		[=](id<1> wiID) {
      accC[wiID] = accA[wiID] + accB[wiID];
    });
  });
}

int main(int argc, char *argv[]) {
  if (argc != 2)
    return 1;

  queue Queue;
  auto D = Queue.get_device();
  if (D.get_info<info::device::host_unified_memory>())
    std::cerr << "Integrated GPU will use zeMemAllocHost\n";
  else
    std::cerr << "Discrete GPU will use zeMemAllocDevice\n";

  if (argv[1][0] == 'u') {
    // Try USM APIs
    // This will generate one each of zeMemAllocHost/Device/Shared
    std::cerr << "Direct USM\n";
    direct_usm(Queue);
  }
  if (argv[1][0] == 's') {
    // Try small buffers
    // This will generate one zeMemAllocHost on Integrated or one
    // zeMemAllocDevice on Discrete GPU
    std::cerr << "Small buffers\n";
    sycl_buffer<int, 4>(Queue);
  }
  if (argv[1][0] == 'l') {
    // Try large buffers
    // This will generate three zeMemAllocHost on Integrated or three
    // zeMemAllocDevice on Discrete GPU
    std::cerr << "Large buffers\n";
    sycl_buffer<long, 10000>(Queue);
  }

  return 0;
}

// Account for 4 extra allocations made by UMF to query page sizes

// CHECK-USM: GPU will use {{zeMemAllocHost|zeMemAllocDevice}}
// CHECK-USM: zeMemAllocDevice = 2
// CHECK-USM:   zeMemAllocHost = 2
// CHECK-USM: zeMemAllocShared = 3
// CHECK-USM-SAME:   zeMemFree = 7

// CHECK-SMALL-BUF: GPU will use [[API:zeMemAllocHost|zeMemAllocDevice]]
// CHECK-SMALL-BUF: zeMemAllocDevice = {{1|2}}
// CHECK-SMALL-BUF: zeMemAllocHost = {{1|2}}
// CHECK-SMALL-BUF: zeMemAllocShared = 2
// CHECK-SMALL-BUF-SAME: zeMemFree = 5

// CHECK-LARGE-BUF: GPU will use [[API:zeMemAllocHost|zeMemAllocDevice]]
// CHECK-LARGE-BUF: zeMemAllocDevice = {{1|4}}
// CHECK-LARGE-BUF: zeMemAllocHost = {{1|4}}
// CHECK-LARGE-BUF: zeMemAllocShared = 2
// CHECK-LARGE-BUF-SAME: zeMemFree = 7
