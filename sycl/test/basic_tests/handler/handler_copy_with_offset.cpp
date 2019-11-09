// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==--- handler_copy_with_offset.cpp - SYCL handler copy with offset test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <cassert>
#include <exception>
#include <memory>
#include <numeric>
#include <vector>

using namespace cl::sycl;
constexpr access::mode read = access::mode::read;
constexpr access::mode write = access::mode::write;
constexpr access::target global_buffer = access::target::global_buffer;

int main() {
  {
    constexpr size_t Size = 8;

    vector_class<char> DataRaw(Size, 'x');
    {
      buffer<char> Buffer{DataRaw.data(), range<1>{Size}};

      vector_class<char> DataGold(Size);
      std::iota(DataGold.begin(), DataGold.end(), '0');

      queue Queue;
      Queue.submit([&](handler &CGH) {
        range<1> AccessRange{4};
        id<1> AccessOffset{2};
        auto Accessor = Buffer.get_access<write, global_buffer>(
            CGH, AccessRange, AccessOffset);
        CGH.copy(DataGold.data(), Accessor);
      });
      Queue.wait();
    }

    vector_class<char> Expected{'x', 'x', '0', '1', '2', '3', 'x', 'x'};
    if (DataRaw != Expected)
      throw std::runtime_error("Check of hadler.copy(ptr, acc) was failed");
  }

  {
    constexpr size_t Size = 8;
    vector_class<char> DataRaw(Size, 'x');
    {
      vector_class<char> DataGold(Size);
      std::iota(DataGold.begin(), DataGold.end(), '0');
      buffer<char> Buffer{DataGold.data(), range<1>{Size}};

      queue Queue;
      Queue.submit([&](handler &CGH) {
        range<1> AccessRange{4};
        id<1> AccessOffset{2};
        auto Accessor = Buffer.get_access<read, global_buffer>(CGH, AccessRange,
                                                               AccessOffset);
        CGH.copy(Accessor, DataRaw.data());
      });
      Queue.wait();
    }
    vector_class<char> Expected{'2', '3', '4', '5', 'x', 'x', 'x', 'x'};
    if (DataRaw != Expected)
      throw std::runtime_error("Check of hadler.copy(acc, ptr) was failed");
  }
  return 0;
}
