// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out

//==------------------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <iostream>
#include <tuple>

using namespace sycl;

constexpr size_t Align = 256;

struct alignas(Align) Aligned {
  int x;
};

int main() {
  device d;
  // Note that get_context() from a default-constructed queue would behave
  // differently. Such a context would include multiple devices and would have
  // to satisfly all their alignment requirement at once, effectively
  // strictening them.
  context ctx{d};
  queue q{ctx, d};

  auto check = [&q](size_t Alignment, auto AllocFn, int Line = __builtin_LINE(),
                    int Case = 0) {
    // First allocation might naturally be over-aligned. Do several of them to
    // do the verification;
    decltype(AllocFn()) Arr[10];
    for (auto *&Elem : Arr)
      Elem = AllocFn();
    for (auto *Ptr : Arr) {
      auto v = reinterpret_cast<uintptr_t>(Ptr);
      if ((v & (Alignment - 1)) != 0) {
        std::cout << "Failed at line " << Line << ", case " << Case
                  << std::endl;
        assert(false && "Not properly aligned!");
        break; // To be used with commented out assert above.
      }
    }
    for (auto *Ptr : Arr)
      free(Ptr, q);
  };

  // The strictest (largest) fundamental alignment of any type is the alignment
  // of max_align_t. This is, however, smaller than the minimal alignment
  // returned by the underlyging runtime as of now.
  constexpr size_t FAlign = alignof(std::max_align_t);

  auto CheckAll = [&](size_t Expected, auto Funcs,
                      int Line = __builtin_LINE()) {
    std::apply(
        [&](auto... Fs) {
          int Case = 0;
          (void)std::initializer_list<int>{
              (check(Expected, Fs, Line, Case++), 0)...};
        },
        Funcs);
  };

  auto MDevice = [&](auto... args) {
    return malloc_device(sizeof(std::max_align_t), args...);
  };
  CheckAll(FAlign,
           std::tuple{[&]() { return MDevice(q); },
                      [&]() { return MDevice(d, ctx); },
                      [&]() { return MDevice(q, property_list{}); },
                      [&]() { return MDevice(d, ctx, property_list{}); }});

  auto MHost = [&](auto... args) {
    return malloc_host(sizeof(std::max_align_t), args...);
  };
  CheckAll(FAlign,
           std::tuple{[&]() { return MHost(q); }, [&]() { return MHost(ctx); },
                      [&]() { return MHost(q, property_list{}); },
                      [&]() { return MHost(ctx, property_list{}); }});

  if (d.has(aspect::usm_shared_allocations)) {
    auto MShared = [&](auto... args) {
      return malloc_shared(sizeof(std::max_align_t), args...);
    };

    CheckAll(FAlign,
             std::tuple{[&]() { return MShared(q); },
                        [&]() { return MShared(d, ctx); },
                        [&]() { return MShared(q, property_list{}); },
                        [&]() { return MShared(d, ctx, property_list{}); }});
  }

  auto ADevice = [&](size_t Align, auto... args) {
    return aligned_alloc_device(Align, sizeof(std::max_align_t), args...);
  };
  CheckAll(FAlign,
           std::tuple{
               [&]() { return ADevice(FAlign / 2, q); },
               [&]() { return ADevice(FAlign / 2, d, ctx); },
               [&]() { return ADevice(FAlign / 2, q, property_list{}); },
               [&]() { return ADevice(FAlign / 2, d, ctx, property_list{}); }});
  CheckAll(
      Align,
      std::tuple{[&]() { return ADevice(Align, q); },
                 [&]() { return ADevice(Align, d, ctx); },
                 [&]() { return ADevice(Align, q, property_list{}); },
                 [&]() { return ADevice(Align, d, ctx, property_list{}); }});

  auto AHost = [&](size_t Align, auto... args) {
    return aligned_alloc_host(Align, sizeof(std::max_align_t), args...);
  };
  CheckAll(
      FAlign,
      std::tuple{[&]() { return AHost(FAlign / 2, q); },
                 [&]() { return AHost(FAlign / 2, ctx); },
                 [&]() { return AHost(FAlign / 2, q, property_list{}); },
                 [&]() { return AHost(FAlign / 2, ctx, property_list{}); }});
  CheckAll(Align,
           std::tuple{[&]() { return AHost(Align, q); },
                      [&]() { return AHost(Align, ctx); },
                      [&]() { return AHost(Align, q, property_list{}); },
                      [&]() { return AHost(Align, ctx, property_list{}); }});

  if (d.has(aspect::usm_shared_allocations)) {
    auto AShared = [&](size_t Align, auto... args) {
      return aligned_alloc_shared(Align, sizeof(std::max_align_t), args...);
    };
    CheckAll(
        FAlign,
        std::tuple{
            [&]() { return AShared(FAlign / 2, q); },
            [&]() { return AShared(FAlign / 2, d, ctx); },
            [&]() { return AShared(FAlign / 2, q, property_list{}); },
            [&]() { return AShared(FAlign / 2, d, ctx, property_list{}); }});
    CheckAll(
        Align,
        std::tuple{[&]() { return AShared(Align, q); },
                   [&]() { return AShared(Align, d, ctx); },
                   [&]() { return AShared(Align, q, property_list{}); },
                   [&]() { return AShared(Align, d, ctx, property_list{}); }});
  }

  auto TDevice = [&](auto... args) {
    return malloc_device<Aligned>(1, args...);
  };
  CheckAll(Align, std::tuple{[&]() { return TDevice(q); },
                             [&]() { return TDevice(d, ctx); }});

  auto THost = [&](auto... args) { return malloc_host<Aligned>(1, args...); };
  CheckAll(Align, std::tuple{[&]() { return THost(q); },
                             [&]() { return THost(ctx); }});

  if (d.has(aspect::usm_shared_allocations)) {
    auto TShared = [&](auto... args) {
      return malloc_shared<Aligned>(1, args...);
    };
    CheckAll(Align, std::tuple{[&]() { return TShared(q); },
                               [&]() { return TShared(d, ctx); }});
  }

  auto ATDevice = [&](size_t Align, auto... args) {
    return aligned_alloc_device<Aligned>(Align, 1, args...);
  };
  CheckAll(Align, std::tuple{[&]() { return ATDevice(Align / 2, q); },
                             [&]() { return ATDevice(Align / 2, d, ctx); }});
  CheckAll(Align * 2,
           std::tuple{[&]() { return ATDevice(Align * 2, q); },
                      [&]() { return ATDevice(Align * 2, d, ctx); }});

  auto ATHost = [&](size_t Align, auto... args) {
    return aligned_alloc_host<Aligned>(Align, 1, args...);
  };
  CheckAll(Align, std::tuple{[&]() { return ATHost(Align / 2, q); },
                             [&]() { return ATHost(Align / 2, ctx); }});
  CheckAll(Align * 2, std::tuple{[&]() { return ATHost(Align * 2, q); },
                                 [&]() { return ATHost(Align * 2, ctx); }});
  if (d.has(aspect::usm_shared_allocations)) {
    auto ATShared = [&](size_t Align, auto... args) {
      return aligned_alloc_shared<Aligned>(Align, 1, args...);
    };
    CheckAll(Align, std::tuple{[&]() { return ATShared(Align / 2, q); },
                               [&]() { return ATShared(Align / 2, d, ctx); }});
    CheckAll(Align * 2,
             std::tuple{[&]() { return ATShared(Align * 2, q); },
                        [&]() { return ATShared(Align * 2, d, ctx); }});
  }

  auto Malloc = [&](auto... args) {
    return malloc(sizeof(std::max_align_t), args...);
  };
  CheckAll(
      FAlign,
      std::tuple{
          [&]() { return Malloc(q, usm::alloc::host); },
          [&]() { return Malloc(d, ctx, usm::alloc::host); },
          [&]() { return Malloc(q, usm::alloc::host, property_list{}); },
          [&]() { return Malloc(d, ctx, usm::alloc::host, property_list{}); }});

  auto TMalloc = [&](auto... args) { return malloc<Aligned>(1, args...); };
  CheckAll(Align,
           std::tuple{[&]() { return TMalloc(q, usm::alloc::host); },
                      [&]() { return TMalloc(d, ctx, usm::alloc::host); }});

  auto AMalloc = [&](size_t Align, auto... args) {
    return aligned_alloc(Align, sizeof(std::max_align_t), args...);
  };
  CheckAll(FAlign,
           std::tuple{
               [&]() { return AMalloc(FAlign / 2, q, usm::alloc::host); },
               [&]() { return AMalloc(FAlign / 2, d, ctx, usm::alloc::host); },
               [&]() {
                 return AMalloc(FAlign / 2, q, usm::alloc::host,
                                property_list{});
               },
               [&]() {
                 return AMalloc(FAlign / 2, d, ctx, usm::alloc::host,
                                property_list{});
               }});
  CheckAll(
      Align,
      std::tuple{[&]() { return AMalloc(Align, q, usm::alloc::host); },
                 [&]() { return AMalloc(Align, d, ctx, usm::alloc::host); },
                 [&]() {
                   return AMalloc(Align, q, usm::alloc::host, property_list{});
                 },
                 [&]() {
                   return AMalloc(Align, d, ctx, usm::alloc::host,
                                  property_list{});
                 }});

  auto ATMalloc = [&](size_t Align, auto... args) {
    return aligned_alloc<Aligned>(Align, 1, args...);
  };
  CheckAll(
      Align,
      std::tuple{
          [&]() { return ATMalloc(Align / 2, q, usm::alloc::host); },
          [&]() { return ATMalloc(Align / 2, d, ctx, usm::alloc::host); }});
  CheckAll(
      Align * 2,
      std::tuple{
          [&]() { return ATMalloc(Align * 2, q, usm::alloc::host); },
          [&]() { return ATMalloc(Align * 2, d, ctx, usm::alloc::host); }});

  return 0;
}
