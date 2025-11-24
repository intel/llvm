// REQUIRES: gpu
// RUN: %{build} -o %t1.out
// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE=1 %{run} %t1.out | FileCheck %s --check-prefix=CHECK-DEFAULT

// RUN: %{build} -fsycl-range-rounding=force -o %t2.out
// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE=1 %{run} %t2.out | FileCheck %s --check-prefix=CHECK-DEFAULT

// RUN: %{build} -fsycl-exp-range-rounding -o %t3.out
// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE=1 %{run} %t3.out | FileCheck %s --check-prefix=CHECK-EXP

// RUN: %{build} -fsycl-range-rounding=force -fsycl-exp-range-rounding -o %t4.out
// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE=1 %{run} %t4.out | FileCheck %s --check-prefix=CHECK-EXP
//
// These tests test 3 things:
//
// 1. The user range is the same as the in kernel range (using RangePtr) as
//    reported by get_range().
// 2. That the effective range is the same as the reported range (using
//    CouterPtr). i.e. check that the mapping of effective range to user range
//    is "onto".
// 3. That every index in a 1, 2, or 3 dimension range is active the execution
//    (using ItemIndexesPtr). i.e. check that the mapping of effective range to
//    user range is "one-to-one".
//
// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/17077
//
#include <sycl/atomic_ref.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>
#include <sycl/vector.hpp>

#include "../helpers.hpp"

#include <iostream>

using namespace sycl;

constexpr size_t MagicY = 33, MagicZ = 64;

range<1> Range1 = {0};
range<2> Range2 = {0, 0};
range<3> Range3 = {0, 0, 0};

template <typename T> class Kernel1;
template <typename T> class Kernel2;
template <typename T> class Kernel3;

void check(const char *msg, size_t v, size_t ref) {
  std::cout << msg << v << std::endl;
  assert(v == ref);
}

template <unsigned Dims> void checkVec(vec<int, Dims> a, vec<int, Dims> b) {
  static_assert(Dims == 1 || Dims == 2 || Dims == 3,
                "Should only be use for 1, 2 or 3 dimensional vectors");
  assert(a[0] == b[0]);
  if constexpr (Dims > 1)
    assert(a[1] == b[1]);
  if constexpr (Dims > 2)
    assert(a[2] == b[2]);
}

template <typename KernelIdT>
void try_1d_range(size_t size, bool useShortcutFunction) {
  using IndexCheckT = int;
  range<1> Range{size};
  queue Queue;

  range<1> *RangePtr = malloc_shared<range<1>>(1, Queue);
  int *CounterPtr = malloc_shared<int>(1, Queue);
  IndexCheckT *ItemIndexesPtr = malloc_shared<IndexCheckT>(Range[0], Queue);

  auto KernelFunc = [=](KernelIdT I) {
    auto atm = atomic_ref<int, sycl::memory_order::relaxed,
                          sycl::memory_scope::device>(*CounterPtr);
    atm.fetch_add(1);
    if constexpr (std::is_same_v<KernelIdT, item<1>>)
      (*RangePtr) = range<1>(I.get_range(0));
    int Idx = I[0];
    ItemIndexesPtr[Idx] = IndexCheckT(I[0]);
  };

  command_submit_wrappers::parallel_for_wrapper<Kernel1<KernelIdT>>(
      useShortcutFunction, Queue, Range, KernelFunc);

  Queue.wait();

  if constexpr (std::is_same_v<KernelIdT, item<1>>) {
    check("Size seen by user at Dim 0 = ", RangePtr->get(0), size);
  }
  check("Counter = ", *CounterPtr, size);
  for (auto i = 0; i < Range[0]; ++i) {
    checkVec<1>(vec<int, 1>(ItemIndexesPtr[i]), vec<int, 1>(i));
  }
  std::cout << "Correct kernel indexes used\n";

  auto Context = Queue.get_context();
  free(RangePtr, Context);
  free(CounterPtr, Context);
  free(ItemIndexesPtr, Context);
}

template <typename KernelIdT>
void try_2d_range(size_t size, bool useShortcutFunction) {
  using IndexCheckT = int2;
  range<2> Range{size, MagicY};
  queue Queue;

  range<2> *RangePtr = malloc_shared<range<2>>(1, Queue);
  int *CounterPtr = malloc_shared<int>(1, Queue);
  IndexCheckT *ItemIndexesPtr =
      malloc_shared<IndexCheckT>(Range[0] * Range[1], Queue);

  auto KernelFunc = [=](KernelIdT I) {
    auto atm = atomic_ref<int, sycl::memory_order::relaxed,
                          sycl::memory_scope::device>(*CounterPtr);
    atm.fetch_add(1);

    if constexpr (std::is_same_v<KernelIdT, item<2>>)
      (*RangePtr) = range<2>(I.get_range(0), I.get_range(1));
    int Idx = I[0] * Range[1] + I[1];
    ItemIndexesPtr[Idx] = IndexCheckT(I[0], I[1]);
  };

  command_submit_wrappers::parallel_for_wrapper<Kernel2<KernelIdT>>(
      useShortcutFunction, Queue, Range, KernelFunc);

  Queue.wait();

  if constexpr (std::is_same_v<KernelIdT, item<2>>) {
    check("Size seen by user at Dim 0 = ", RangePtr->get(0), Range[0]);
    check("Size seen by user at Dim 1 = ", RangePtr->get(1), Range[1]);
  }
  check("Counter = ", *CounterPtr, size * MagicY);
  for (auto i = 0; i < Range[0]; ++i)
    for (auto j = 0; j < Range[1]; ++j)
      checkVec<2>(ItemIndexesPtr[i * Range[1] + j], IndexCheckT(i, j));
  std::cout << "Correct kernel indexes used\n";

  auto Context = Queue.get_context();
  free(RangePtr, Context);
  free(CounterPtr, Context);
  free(ItemIndexesPtr, Context);
}

template <typename KernelIdT>
void try_3d_range(size_t size, bool useShortcutFunction) {
  using IndexCheckT = int3;
  range<3> Range{size, MagicY, MagicZ};
  queue Queue;

  range<3> *RangePtr = malloc_shared<range<3>>(1, Queue);
  int *CounterPtr = malloc_shared<int>(1, Queue);
  IndexCheckT *ItemIndexesPtr =
      malloc_shared<IndexCheckT>(Range[0] * Range[1] * Range[2], Queue);

  auto KernelFunc = [=](KernelIdT I) {
    auto atm = atomic_ref<int, sycl::memory_order::relaxed,
                          sycl::memory_scope::device>(*CounterPtr);
    atm.fetch_add(1);

    if constexpr (std::is_same_v<KernelIdT, item<3>>)
      (*RangePtr) = range<3>(I.get_range(0), I.get_range(1), I.get_range(2));
    int Idx = I[0] * Range[1] * Range[2] + I[1] * Range[2] + I[2];
    ItemIndexesPtr[Idx] = IndexCheckT(I[0], I[1], I[2]);
  };

  command_submit_wrappers::parallel_for_wrapper<Kernel2<KernelIdT>>(
      useShortcutFunction, Queue, Range, KernelFunc);

  Queue.wait();

  if constexpr (std::is_same_v<KernelIdT, item<3>>) {
    check("Size seen by user at Dim 0 = ", RangePtr->get(0), Range[0]);
    check("Size seen by user at Dim 1 = ", RangePtr->get(1), Range[1]);
    check("Size seen by user at Dim 2 = ", RangePtr->get(2), Range[2]);
  }
  check("Counter = ", *CounterPtr, size * MagicY * MagicZ);
  for (auto i = 0; i < Range[0]; ++i)
    for (auto j = 0; j < Range[1]; ++j)
      for (auto k = 0; k < Range[2]; ++k)
        checkVec<3>(ItemIndexesPtr[i * Range[1] * Range[2] + j * Range[2] + k],
                    IndexCheckT(i, j, k));
  std::cout << "Correct kernel indexes used\n";

  auto Context = Queue.get_context();
  free(RangePtr, Context);
  free(CounterPtr, Context);
  free(ItemIndexesPtr, Context);
}

void try_unnamed_lambda(size_t size, bool useShortcutFunction) {
  range<3> Range{size, MagicY, MagicZ};
  queue Queue;

  range<3> *RangePtr = malloc_shared<range<3>>(1, Queue);
  int *CounterPtr = malloc_shared<int>(1, Queue);

  auto KernelFunc = [=](id<3> ID) {
    auto atm = atomic_ref<int, sycl::memory_order::relaxed,
                          sycl::memory_scope::device>(*CounterPtr);
    atm.fetch_add(1);
    (*RangePtr)[0] = ID[0];
  };

  command_submit_wrappers::parallel_for_wrapper<class TestKernel>(
      useShortcutFunction, Queue, Range, KernelFunc);

  Queue.wait();

  check("Counter = ", *CounterPtr, size * MagicY * MagicZ);

  auto Context = Queue.get_context();
  free(RangePtr, Context);
  free(CounterPtr, Context);
}

int main() {
  int x = 1500;
  try_1d_range<item<1>>(x, true);
  try_1d_range<id<1>>(x, true);
  try_2d_range<item<2>>(x, true);
  try_2d_range<id<2>>(x, true);
  try_3d_range<item<3>>(x, true);
  try_3d_range<id<3>>(x, true);
  try_unnamed_lambda(x, true);

  try_1d_range<item<1>>(x, false);
  try_1d_range<id<1>>(x, false);
  try_2d_range<item<2>>(x, false);
  try_2d_range<id<2>>(x, false);
  try_3d_range<item<3>>(x, false);
  try_3d_range<id<3>>(x, false);
  try_unnamed_lambda(x, false);

  x = 256;
  try_1d_range<item<1>>(x, true);
  try_1d_range<id<1>>(x, true);
  try_2d_range<item<2>>(x, true);
  try_2d_range<id<2>>(x, true);
  try_3d_range<item<3>>(x, true);
  try_3d_range<id<3>>(x, true);
  try_unnamed_lambda(x, true);

  try_1d_range<item<1>>(x, false);
  try_1d_range<id<1>>(x, false);
  try_2d_range<item<2>>(x, false);
  try_2d_range<id<2>>(x, false);
  try_3d_range<item<3>>(x, false);
  try_3d_range<id<3>>(x, false);
  try_unnamed_lambda(x, false);
}

// CHECK-DEFAULT:       parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 0 = 1500
// CHECK-DEFAULT-NEXT:  Counter = 1500
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-DEFAULT-NEXT:  Counter = 1500
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 0 = 1500
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 1 = 33
// CHECK-DEFAULT-NEXT:  Counter = 49500
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-DEFAULT-NEXT:  Counter = 49500
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 0 = 1500
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 1 = 33
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 2 = 64
// CHECK-DEFAULT-NEXT:  Counter = 3168000
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-DEFAULT-NEXT:  Counter = 3168000
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-DEFAULT-NEXT:  Counter = 3168000
// CHECK-DEFAULT:       parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 0 = 1500
// CHECK-DEFAULT-NEXT:  Counter = 1500
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-DEFAULT-NEXT:  Counter = 1500
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 0 = 1500
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 1 = 33
// CHECK-DEFAULT-NEXT:  Counter = 49500
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-DEFAULT-NEXT:  Counter = 49500
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 0 = 1500
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 1 = 33
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 2 = 64
// CHECK-DEFAULT-NEXT:  Counter = 3168000
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-DEFAULT-NEXT:  Counter = 3168000
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-DEFAULT-NEXT:  Counter = 3168000
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 0 = 256
// CHECK-DEFAULT-NEXT:  Counter = 256
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  Counter = 256
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 0 = 256
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 1 = 33
// CHECK-DEFAULT-NEXT:  Counter = 8448
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  Counter = 8448
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 0 = 256
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 1 = 33
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 2 = 64
// CHECK-DEFAULT-NEXT:  Counter = 540672
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  Counter = 540672
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  Counter = 540672
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 0 = 256
// CHECK-DEFAULT-NEXT:  Counter = 256
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  Counter = 256
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 0 = 256
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 1 = 33
// CHECK-DEFAULT-NEXT:  Counter = 8448
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  Counter = 8448
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 0 = 256
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 1 = 33
// CHECK-DEFAULT-NEXT:  Size seen by user at Dim 2 = 64
// CHECK-DEFAULT-NEXT:  Counter = 540672
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  Counter = 540672
// CHECK-DEFAULT-NEXT:  Correct kernel indexes used
// CHECK-DEFAULT-NEXT:  Counter = 540672

// CHECK-EXP:       parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-EXP-NEXT:  Size seen by user at Dim 0 = 1500
// CHECK-EXP-NEXT:  Counter = 1500
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-EXP-NEXT:  Counter = 1500
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 48
// CHECK-EXP-NEXT:  Size seen by user at Dim 0 = 1500
// CHECK-EXP-NEXT:  Size seen by user at Dim 1 = 33
// CHECK-EXP-NEXT:  Counter = 49500
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 48
// CHECK-EXP-NEXT:  Counter = 49500
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 40
// CHECK-EXP-NEXT:  Size seen by user at Dim 0 = 1500
// CHECK-EXP-NEXT:  Size seen by user at Dim 1 = 33
// CHECK-EXP-NEXT:  Size seen by user at Dim 2 = 64
// CHECK-EXP-NEXT:  Counter = 3168000
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 40
// CHECK-EXP-NEXT:  Counter = 3168000
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 40
// CHECK-EXP-NEXT:  Counter = 3168000
// CHECK-EXP:       parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-EXP-NEXT:  Size seen by user at Dim 0 = 1500
// CHECK-EXP-NEXT:  Counter = 1500
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-EXP-NEXT:  Counter = 1500
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 48
// CHECK-EXP-NEXT:  Size seen by user at Dim 0 = 1500
// CHECK-EXP-NEXT:  Size seen by user at Dim 1 = 33
// CHECK-EXP-NEXT:  Counter = 49500
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 48
// CHECK-EXP-NEXT:  Counter = 49500
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 40
// CHECK-EXP-NEXT:  Size seen by user at Dim 0 = 1500
// CHECK-EXP-NEXT:  Size seen by user at Dim 1 = 33
// CHECK-EXP-NEXT:  Size seen by user at Dim 2 = 64
// CHECK-EXP-NEXT:  Counter = 3168000
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 40
// CHECK-EXP-NEXT:  Counter = 3168000
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 0 from 1500 to 1504
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 40
// CHECK-EXP-NEXT:  Counter = 3168000
// CHECK-EXP-NEXT:  Size seen by user at Dim 0 = 256
// CHECK-EXP-NEXT:  Counter = 256
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  Counter = 256
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 48
// CHECK-EXP-NEXT:  Size seen by user at Dim 0 = 256
// CHECK-EXP-NEXT:  Size seen by user at Dim 1 = 33
// CHECK-EXP-NEXT:  Counter = 8448
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 48
// CHECK-EXP-NEXT:  Counter = 8448
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 40
// CHECK-EXP-NEXT:  Size seen by user at Dim 0 = 256
// CHECK-EXP-NEXT:  Size seen by user at Dim 1 = 33
// CHECK-EXP-NEXT:  Size seen by user at Dim 2 = 64
// CHECK-EXP-NEXT:  Counter = 540672
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 40
// CHECK-EXP-NEXT:  Counter = 540672
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 40
// CHECK-EXP-NEXT:  Counter = 540672
// CHECK-EXP-NEXT:  Size seen by user at Dim 0 = 256
// CHECK-EXP-NEXT:  Counter = 256
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  Counter = 256
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 48
// CHECK-EXP-NEXT:  Size seen by user at Dim 0 = 256
// CHECK-EXP-NEXT:  Size seen by user at Dim 1 = 33
// CHECK-EXP-NEXT:  Counter = 8448
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 48
// CHECK-EXP-NEXT:  Counter = 8448
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 40
// CHECK-EXP-NEXT:  Size seen by user at Dim 0 = 256
// CHECK-EXP-NEXT:  Size seen by user at Dim 1 = 33
// CHECK-EXP-NEXT:  Size seen by user at Dim 2 = 64
// CHECK-EXP-NEXT:  Counter = 540672
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 40
// CHECK-EXP-NEXT:  Counter = 540672
// CHECK-EXP-NEXT:  Correct kernel indexes used
// CHECK-EXP-NEXT:  parallel_for range adjusted at dim 1 from 33 to 40
// CHECK-EXP-NEXT:  Counter = 540672
