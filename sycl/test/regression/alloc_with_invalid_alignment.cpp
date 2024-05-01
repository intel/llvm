// REQUIRES: level_zero
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==----- alloc_with_invalid_alignment.cpp - SYCL USM allocation test-----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace ext::oneapi::experimental;
using namespace ext::intel::experimental;
using alloc = usm::alloc;

template <typename T> void testAlign(sycl::queue &q, unsigned align) {
  const sycl::context &Ctx = q.get_context();
  auto dev = q.get_device();

  constexpr int N = 10;
  assert(align > 0 || (align & (align - 1)) == 0);
  auto ADevice = [&](size_t align, auto... args) {
    return aligned_alloc_device(align, N, args...);
  };
  auto AHost = [&](size_t align, auto... args) {
    return aligned_alloc_host(align, N, args...);
  };
  auto AShared = [&](size_t align, auto... args) {
    return aligned_alloc_shared(align, N, args...);
  };
  auto AAnnotated = [&](size_t align, auto... args) {
    return aligned_alloc(align, N, args...);
  };

  // Test cases that are expected to return null
  auto check_null = [&q](auto AllocFn, int Line, int Case) {
    decltype(AllocFn()) Ptr = AllocFn();
    if (Ptr != nullptr) {
      free(Ptr, q);
      std::cout << "Failed at line " << Line << ", case " << Case << std::endl;
      assert(false && "The return is not null!");
    }
  };

  auto CheckNullAll = [&](auto Funcs, int Line = __builtin_LINE()) {
    std::apply(
        [&](auto... Fs) {
          int Case = 0;
          (void)std::initializer_list<int>{
              (check_null(Fs, Line, Case++), 0)...};
        },
        Funcs);
  };
  CheckNullAll(std::tuple{
      // Case: aligned_alloc_xxx with no alignment property, and the alignment
      // argument is not a power of 2, the result is nullptr
      [&]() { return ADevice(3, q); }, [&]() { return ADevice(5, dev, Ctx); },
      [&]() { return AHost(7, q); }, [&]() { return AHost(9, Ctx); },
      [&]() { return AShared(114, q); },
      [&]() { return AShared(1023, dev, Ctx); },
      [&]() { return AAnnotated(15, q, alloc::device); },
      [&]() { return AAnnotated(17, dev, Ctx, alloc::host); }});
}

int main() {
  sycl::queue q;
  testAlign<char>(q, 4);
  return 0;
}
