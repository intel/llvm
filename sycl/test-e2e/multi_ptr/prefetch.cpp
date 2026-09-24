// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//===------------------------- prefetch.cpp
//--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cmath>
#include <iostream>
#include <sycl/multi_ptr.hpp>
#include <sycl/queue.hpp>
#include <vector>

using namespace sycl;

template <access::decorated IsDecorated> class PrefetchKernel;

template <access::decorated IsDecorated> void testPrefetchWithDecoration() {
  constexpr size_t Size = 1024;
  std::vector<int> HostData(Size);
  for (size_t i = 0; i < Size; ++i) {
    HostData[i] = static_cast<int>(i);
  }

  queue Q;
  buffer<int, 1> Buf(HostData.data(), range<1>(Size));

  Q.submit([&](handler &CGH) {
    auto Acc = Buf.get_access<access::mode::read_write>(CGH);

    CGH.parallel_for<PrefetchKernel<IsDecorated>>(
        range<1>(Size), [=](id<1> Index) {
          auto Ptr = Acc.template get_multi_ptr<IsDecorated>();

          // Test prefetch with different element counts
          if (Index[0] == 0) {
            Ptr.prefetch(1);
            Ptr.prefetch(16);
            Ptr.prefetch(64);
            Ptr.prefetch(256);
          }

          // Test prefetch at different offsets
          if (Index[0] < Size - 100) {
            auto OffsetPtr = Ptr + Index[0];
            OffsetPtr.prefetch(10);
          }

          // Actual computation to ensure prefetch is useful
          int Sum = 0;
          for (size_t i = 0; i < 10 && Index[0] + i < Size; ++i) {
            Sum += Ptr[Index[0] + i];
          }
          Acc[Index] = Sum;
        });
  });

  Q.wait();

  // Verify results
  auto HostAcc = Buf.get_host_access();
  for (size_t i = 0; i < Size; ++i) {
    int Expected = 0;
    for (size_t j = 0; j < 10 && i + j < Size; ++j) {
      Expected += static_cast<int>(i + j);
    }
    assert(HostAcc[i] == Expected && "Prefetch test failed");
  }
}

void testPrefetchWithGlobalPointer() {
  constexpr size_t Size = 512;
  std::vector<float> HostData(Size);
  for (size_t i = 0; i < Size; ++i) {
    HostData[i] = static_cast<float>(i) * 0.5f;
  }

  queue Q;
  buffer<float, 1> Buf(HostData.data(), range<1>(Size));

  Q.submit([&](handler &CGH) {
    auto Acc = Buf.get_access<access::mode::read_write>(CGH);

    CGH.parallel_for<class GlobalPrefetchKernel>(
        range<1>(Size), [=](id<1> Index) {
          using global_ptr =
              multi_ptr<float, access::address_space::global_space,
                        access::decorated::yes>;

          global_ptr Ptr =
              address_space_cast<access::address_space::global_space,
                                 access::decorated::yes>(&Acc[0]);

          // Prefetch future data
          if (Index[0] < Size - 50) {
            auto FuturePtr = Ptr + Index[0] + 10;
            FuturePtr.prefetch(20);
          }

          // Process data
          float Sum = 0.0f;
          for (size_t i = 0; i < 5 && Index[0] + i < Size; ++i) {
            Sum += Ptr[Index[0] + i];
          }
          Acc[Index] = Sum;
        });
  });

  Q.wait();

  // Verify results
  auto HostAcc = Buf.get_host_access();
  for (size_t i = 0; i < Size; ++i) {
    float Expected = 0.0f;
    for (size_t j = 0; j < 5 && i + j < Size; ++j) {
      Expected += static_cast<float>(i + j) * 0.5f;
    }
    assert(std::abs(HostAcc[i] - Expected) < 0.001f &&
           "Global prefetch test failed");
  }
}

void testPrefetchWithLargeData() {
  constexpr size_t Size = 4096;
  std::vector<double> HostData(Size);
  for (size_t i = 0; i < Size; ++i) {
    HostData[i] = static_cast<double>(i);
  }

  queue Q;
  buffer<double, 1> Buf(HostData.data(), range<1>(Size));

  Q.submit([&](handler &CGH) {
    auto Acc = Buf.get_access<access::mode::read>(CGH);

    CGH.parallel_for<class LargePrefetchKernel>(
        range<1>(Size / 8), [=](id<1> Index) {
          auto Ptr = Acc.template get_multi_ptr<access::decorated::yes>();
          size_t BaseIndex = Index[0] * 8;

          // Prefetch a chunk of data
          auto ChunkPtr = Ptr + BaseIndex;
          ChunkPtr.prefetch(64);

          // Process the prefetched chunk
          double Sum = 0.0;
          for (size_t i = 0; i < 8; ++i) {
            Sum += ChunkPtr[i];
          }
        });
  });

  Q.wait();

  auto HostAcc = Buf.get_host_access();
  for (size_t i = 0; i < Size; ++i) {
    assert(HostAcc[i] == static_cast<double>(i) &&
           "Large prefetch test failed");
  }
}

void testPrefetchAtBoundaries() {
  constexpr size_t Size = 256;
  std::vector<int> HostData(Size, 42);

  queue Q;
  buffer<int, 1> Buf(HostData.data(), range<1>(Size));

  Q.submit([&](handler &CGH) {
    auto Acc = Buf.get_access<access::mode::read>(CGH);

    CGH.single_task<class BoundaryPrefetchKernel>([=]() {
      auto Ptr = Acc.template get_multi_ptr<access::decorated::no>();

      // Prefetch at start
      Ptr.prefetch(1);
      Ptr.prefetch(10);

      // Prefetch at end (careful not to go beyond buffer)
      auto EndPtr = Ptr + (Size - 10);
      EndPtr.prefetch(10);

      // Prefetch zero elements (edge case)
      Ptr.prefetch(0);
    });
  });

  Q.wait();

  auto HostAcc = Buf.get_host_access();
  for (size_t i = 0; i < Size; ++i) {
    assert(HostAcc[i] == 42 && "Boundary prefetch test failed");
  }
}

void testPrefetchWithStructs() {
  struct TestStruct {
    int A;
    float B;
    double C;
  };

  constexpr size_t Size = 128;
  std::vector<TestStruct> HostData(Size);
  for (size_t i = 0; i < Size; ++i) {
    HostData[i] = {static_cast<int>(i), static_cast<float>(i) * 1.5f,
                   static_cast<double>(i) * 2.5};
  }

  queue Q;
  buffer<TestStruct, 1> Buf(HostData.data(), range<1>(Size));

  Q.submit([&](handler &CGH) {
    auto Acc = Buf.get_access<access::mode::read>(CGH);

    CGH.parallel_for<class StructPrefetchKernel>(
        range<1>(Size / 2), [=](id<1> Index) {
          auto Ptr = Acc.template get_multi_ptr<access::decorated::yes>();
          size_t BaseIdx = Index[0] * 2;

          // Prefetch structures
          auto StructPtr = Ptr + BaseIdx;
          StructPtr.prefetch(8);

          // Access the data
          int Sum = 0;
          for (size_t i = 0; i < 2 && BaseIdx + i < Size; ++i) {
            Sum += StructPtr[i].A;
          }
        });
  });

  Q.wait();

  auto HostAcc = Buf.get_host_access();
  for (size_t i = 0; i < Size; ++i) {
    assert(HostAcc[i].A == static_cast<int>(i) &&
           "Struct prefetch integer field test failed");
    assert(std::abs(HostAcc[i].B - static_cast<float>(i) * 1.5f) < 0.001f &&
           "Struct prefetch float field test failed");
    assert(std::abs(HostAcc[i].C - static_cast<double>(i) * 2.5) < 0.001 &&
           "Struct prefetch double field test failed");
  }
}

int main() {
  // Test prefetch with decorated pointers
  testPrefetchWithDecoration<access::decorated::yes>();
  testPrefetchWithDecoration<access::decorated::no>();

  // Test prefetch with explicit global pointers
  testPrefetchWithGlobalPointer();

  // Test prefetch with large datasets
  testPrefetchWithLargeData();

  // Test prefetch at buffer boundaries
  testPrefetchAtBoundaries();

  // Test prefetch with complex data structures
  testPrefetchWithStructs();

  std::cout << "All prefetch tests passed!" << std::endl;
  return 0;
}
