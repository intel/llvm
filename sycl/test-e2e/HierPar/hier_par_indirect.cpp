//==- hier_par_indirect.cpp --- hierarchical parallelism test for WG scope--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -Wno-error=deprecated-declarations -o %t.out
// RUN: %{run} %t.out

// This test checks correctness of hierarchical kernel execution when the work
// item code is not directly inside work group scope: work group scope code at
// every level of the call chain must run once per work group, and the work item
// code once per work item.

#include <iostream>
#include <sycl/atomic_ref.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/h_item.hpp>
#include <sycl/usm.hpp>

// Number of times work group scope code ran in each function, and the number
// of times the parallel_for_work_item bodies ran.
struct Counts {
  int Kernel = 0;
  int Bar = 0;
  int Foo = 0;
  int PFWI = 0;
};

static void inc(int &C) {
  sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>
      Ref(C);
  Ref.fetch_add(1);
}

void __attribute__((noinline)) foo(sycl::group<1> work_group,
                                   Counts *Counters) {
  inc(Counters->Foo);
  work_group.parallel_for_work_item(
      [&](sycl::h_item<1> index) { inc(Counters->PFWI); });
  inc(Counters->Foo);
}

void __attribute__((noinline)) bar(sycl::group<1> work_group,
                                   Counts *Counters) {
  inc(Counters->Bar);
  foo(work_group, Counters);
  inc(Counters->Bar);
}

constexpr int WGSize = 128;

static int checkOne(const char *Kernel, const char *Field, int Got, int Want) {
  if (Got == Want)
    return 0;
  std::cout << Kernel << ": " << Field << " = " << Got << ", expected " << Want
            << "\n";
  return 1;
}

// Compares the counters with the expected values for a work group and resets
// them for the next kernel.
static int check(sycl::queue &q, Counts *Counters, const char *Name,
                 const Counts &Expected) {
  Counts Host;
  q.memcpy(&Host, Counters, sizeof(Counts)).wait();
  q.memset(Counters, 0, sizeof(Counts)).wait();
  return checkOne(Name, "Kernel", Host.Kernel, Expected.Kernel) +
         checkOne(Name, "Bar", Host.Bar, Expected.Bar) +
         checkOne(Name, "Foo", Host.Foo, Expected.Foo) +
         checkOne(Name, "PFWI", Host.PFWI, Expected.PFWI);
}

int main(int argc, char **argv) {
  sycl::queue q;
  Counts *Counters = sycl::malloc_device<Counts>(1, q);
  q.memset(Counters, 0, sizeof(Counts)).wait();
  int Errors = 0;

  // Try a single indirect call, two indirect calls and an indirect call
  // accompanied by multiple parallel_for_work_item calls in the same work_group
  // scope.
  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for_work_group(sycl::range<1>{1}, sycl::range<1>{WGSize},
                                 ([=](sycl::group<1> wGroup) {
                                   inc(Counters->Kernel);
                                   foo(wGroup, Counters);
                                   inc(Counters->Kernel);
                                 }));
   }).wait();
  Errors += check(q, Counters, "foo", {.Kernel = 2, .Foo = 2, .PFWI = WGSize});
  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for_work_group(
         sycl::range<1>{1}, sycl::range<1>{WGSize},
         ([=](sycl::group<1> wGroup) {
           inc(Counters->Kernel);
           foo(wGroup, Counters); // 1-layer indirect call
           bar(wGroup, Counters); // 2-layer indirect call since bar calls foo
           inc(Counters->Kernel);
         }));
   }).wait();
  Errors += check(q, Counters, "foo+bar",
                  {.Kernel = 2, .Bar = 2, .Foo = 4, .PFWI = 2 * WGSize});
  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for_work_group(
         sycl::range<1>{1}, sycl::range<1>{WGSize},
         ([=](sycl::group<1> wGroup) {
           inc(Counters->Kernel);
           wGroup.parallel_for_work_item(
               [&](sycl::h_item<1> index) { inc(Counters->PFWI); });
           foo(wGroup, Counters);
           wGroup.parallel_for_work_item(
               [&](sycl::h_item<1> index) { inc(Counters->PFWI); });
           inc(Counters->Kernel);
         }));
   }).wait();
  Errors += check(q, Counters, "pfwi+foo+pfwi",
                  {.Kernel = 2, .Foo = 2, .PFWI = 3 * WGSize});
  // Also try an example of a work-group scope variable being used in work item
  // scope
  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for_work_group(sycl::range<1>{1}, sycl::range<1>{WGSize},
                                 ([=](sycl::group<1> wGroup) {
                                   int data;
                                   inc(Counters->Kernel);
                                   foo(wGroup, Counters);
                                   wGroup.parallel_for_work_item(
                                       [&](sycl::h_item<1> index) {
                                         data = 0;
                                         inc(Counters->PFWI);
                                       });
                                   inc(Counters->Kernel);
                                 }));
   }).wait();
  Errors += check(q, Counters, "wg variable",
                  {.Kernel = 2, .Foo = 2, .PFWI = 2 * WGSize});

  sycl::free(Counters, q);
  if (Errors != 0)
    return 1;
  std::cout << "test passed" << std::endl;
  return 0;
}
