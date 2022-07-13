// RUN: %clangxx -fsycl -DSYCL2020_CONFORMANT_APIS -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=opencl %CPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_FILTER=opencl %GPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_FILTER=opencl %ACC_RUN_PLACEHOLDER %t.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=opencl %CPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_FILTER=opencl %GPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_FILTER=opencl %ACC_RUN_PLACEHOLDER %t.out

//==---------- buffer_allocator.cpp - SYCL buffer allocator tests ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests that the default (and explicit with SYCL 2020) buffer_allocator used by
// buffers are as defined by the spec and will allocate the right amount of
// memory on the device.

#include <iostream>
#include <sycl/sycl.hpp>

constexpr size_t NumElems = 67;

template <typename T> class TestKernel;

template <typename T, typename AllocT>
bool checkResult(sycl::buffer<T, 1, AllocT> &Buf, size_t N, std::string &TName,
                 std::string AllocTName) {
  auto HostAcc = Buf.get_host_access();

  bool Success = true;
  for (size_t I = 0; I < NumElems; ++I) {
    if (HostAcc[I] != static_cast<T>(I)) {
      std::cout << "Wrong value was written at index " << I << " for buffer<"
                << TName << ", 1, " << AllocTName << ">" << std::endl;
      Success = false;
    }
  }
  return Success;
}

template <typename T> bool runTest(sycl::queue &Q, std::string TName) {
  sycl::buffer<T, 1> DefaultBuf{NumElems};

#ifdef SYCL2020_CONFORMANT_APIS
  static_assert(std::is_same_v<decltype(DefaultBuf),
                               sycl::buffer<T, 1, sycl::buffer_allocator<T>>>);

  sycl::buffer<T, 1, sycl::buffer_allocator<char>> CharAllocBuf{NumElems};
  sycl::buffer<T, 1, sycl::buffer_allocator<long>> LongAllocBuf{NumElems};
#else
  static_assert(std::is_same_v<decltype(DefaultBuf),
                               sycl::buffer<T, 1, sycl::buffer_allocator>>);
#endif

  Q.submit([&](sycl::handler &CGH) {
     auto DefaultAcc = DefaultBuf.get_access(CGH);
#ifdef SYCL2020_CONFORMANT_APIS
     auto CharAllocAcc = CharAllocBuf.get_access(CGH);
     auto LongAllocAcc = LongAllocBuf.get_access(CGH);
#endif
     CGH.parallel_for<TestKernel<T>>(NumElems, [=](sycl::item<1> It) {
       DefaultAcc[It] = static_cast<T>(It[0]);
#ifdef SYCL2020_CONFORMANT_APIS
       CharAllocAcc[It] = static_cast<T>(It[0]);
       LongAllocAcc[It] = static_cast<T>(It[0]);
#endif
     });
   }).wait();

#ifdef SYCL2020_CONFORMANT_APIS
  return checkResult(DefaultBuf, NumElems, TName,
                     "buffer_allocator<" + TName + ">") &&
         checkResult(CharAllocBuf, NumElems, TName, "buffer_allocator<char>") &&
         checkResult(LongAllocBuf, NumElems, TName, "buffer_allocator<long>");
#else
  return checkResult(DefaultBuf, NumElems, TName, "buffer_allocator");
#endif
}

#define RUN_TEST(TYPE, Q) runTest<TYPE>(Q, #TYPE)

int main() {
  sycl::queue Queue;
  return !(RUN_TEST(char, Queue) && RUN_TEST(unsigned short, Queue) &&
           RUN_TEST(float, Queue) && RUN_TEST(long, Queue));
}

#undef RUN_TEST
