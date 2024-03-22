// REQUIRES: gpu
// RUN: %{build} -o %t.out
// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE=1 %{run} %t.out | FileCheck %s --check-prefix=CHECK-DEFAULT

// RUN: %{build} -fsycl-range-rounding=force -o %t.out
// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE=1 %{run} %t.out | FileCheck %s --check-prefix=CHECK-DEFAULT

// These tests test 3 things:
//
// 1. The user range is the same as the in kernel range (using BufRange) as
//    reported by get_range().
// 2. That the effective range is the same as the reported range (using
//    BufCounter). i.e. check that the mapping of effective range to user range
//    is "onto".
// 3. That every index in a 1, 2, or 3 dimension range is active the execution
//    (using BufIndexes). i.e. check that the mapping of effective range to user
//    range is "one-to-one".
//
#include <iostream>
#include <sycl/detail/core.hpp>

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

template <typename KernelIdT> void try_1d_range(size_t size) {
  using IndexCheckT = int;
  range<1> Size{size};
  int Counter = 0;
  std::vector<IndexCheckT> ItemIndexes(Size[0]);
  {
    buffer<range<1>, 1> BufRange(&Range1, 1);
    buffer<int, 1> BufCounter(&Counter, 1);
    buffer<IndexCheckT, 1> BufIndexes(ItemIndexes);
    queue myQueue;

    myQueue.submit([&](handler &cgh) {
      auto AccRange = BufRange.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);
      auto AccIndexes = BufIndexes.get_access<access::mode::write>(cgh);
      cgh.parallel_for<Kernel1<KernelIdT>>(Size, [=](KernelIdT I) {
        AccCounter[0].fetch_add(1);
        if constexpr (std::is_same_v<KernelIdT, item<1>>)
          AccRange[0] = sycl::range<1>(I.get_range(0));
        int Idx = I[0];
        AccIndexes[Idx] = IndexCheckT(I[0]);
      });
    });
    myQueue.wait();
  }
  if constexpr (std::is_same_v<KernelIdT, item<1>>) {
    check("Size seen by user at Dim 0 = ", Range1.get(0), size);
  }
  check("Counter = ", Counter, size);
  for (auto i = 0; i < Size[0]; ++i) {
    checkVec<1>(vec<int, 1>(ItemIndexes[i]), vec<int, 1>(i));
  }
  std::cout << "Correct kernel indexes used\n";
}

template <typename KernelIdT> void try_2d_range(size_t size) {
  using IndexCheckT = int2;
  range<2> Size{size, MagicY};
  int Counter = 0;
  std::vector<IndexCheckT> ItemIndexes(Size[0] * Size[1]);
  {
    buffer<range<2>, 1> BufRange(&Range2, 1);
    buffer<int, 1> BufCounter(&Counter, 1);
    buffer<IndexCheckT, 1> BufIndexes(ItemIndexes);
    queue myQueue;

    myQueue.submit([&](handler &cgh) {
      auto AccRange = BufRange.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);
      auto AccIndexes = BufIndexes.get_access<access::mode::write>(cgh);
      cgh.parallel_for<Kernel2<KernelIdT>>(Size, [=](KernelIdT I) {
        AccCounter[0].fetch_add(1);
        if constexpr (std::is_same_v<KernelIdT, item<2>>)
          AccRange[0] = sycl::range<2>(I.get_range(0), I.get_range(1));
        int Idx = I[0] * Size[1] + I[1];
        AccIndexes[Idx] = IndexCheckT(I[0], I[1]);
      });
    });
    myQueue.wait();
  }
  if constexpr (std::is_same_v<KernelIdT, item<2>>) {
    check("Size seen by user at Dim 0 = ", Range2.get(0), Size[0]);
    check("Size seen by user at Dim 1 = ", Range2.get(1), Size[1]);
  }
  check("Counter = ", Counter, size * MagicY);
  for (auto i = 0; i < Size[0]; ++i)
    for (auto j = 0; j < Size[1]; ++j)
      checkVec<2>(ItemIndexes[i * Size[1] + j], IndexCheckT(i, j));
  std::cout << "Correct kernel indexes used\n";
}

template <typename KernelIdT> void try_3d_range(size_t size) {
  using IndexCheckT = int3;
  range<3> Size{size, MagicY, MagicZ};
  int Counter = 0;
  std::vector<IndexCheckT> ItemIndexes(Size[0] * Size[1] * Size[2]);
  {
    buffer<range<3>, 1> BufRange(&Range3, 1);
    buffer<int, 1> BufCounter(&Counter, 1);
    buffer<IndexCheckT, 1> BufIndexes(ItemIndexes);
    queue myQueue;

    myQueue.submit([&](handler &cgh) {
      auto AccRange = BufRange.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);
      auto AccIndexes = BufIndexes.get_access<access::mode::write>(cgh);
      cgh.parallel_for<Kernel3<KernelIdT>>(Size, [=](KernelIdT I) {
        AccCounter[0].fetch_add(1);
        if constexpr (std::is_same_v<KernelIdT, item<3>>)
          AccRange[0] =
              sycl::range<3>(I.get_range(0), I.get_range(1), I.get_range(2));
        int Idx = I[0] * Size[1] * Size[2] + I[1] * Size[2] + I[2];
        AccIndexes[Idx] = IndexCheckT(I[0], I[1], I[2]);
      });
    });
    myQueue.wait();
  }
  if constexpr (std::is_same_v<KernelIdT, item<3>>) {
    check("Size seen by user at Dim 0 = ", Range3.get(0), Size[0]);
    check("Size seen by user at Dim 1 = ", Range3.get(1), Size[1]);
    check("Size seen by user at Dim 2 = ", Range3.get(2), Size[2]);
  }
  check("Counter = ", Counter, size * MagicY * MagicZ);
  for (auto i = 0; i < Size[0]; ++i)
    for (auto j = 0; j < Size[1]; ++j)
      for (auto k = 0; k < Size[2]; ++k)
        checkVec<3>(ItemIndexes[i * Size[1] * Size[2] + j * Size[2] + k],
                    IndexCheckT(i, j, k));
  std::cout << "Correct kernel indexes used\n";
}

void try_unnamed_lambda(size_t size) {
  range<3> Size{size, MagicY, MagicZ};
  int Counter = 0;
  {
    buffer<range<3>, 1> BufRange(&Range3, 1);
    buffer<int, 1> BufCounter(&Counter, 1);
    queue myQueue;

    myQueue.submit([&](handler &cgh) {
      auto AccRange = BufRange.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);
      cgh.parallel_for(Size, [=](id<3> ID) {
        AccCounter[0].fetch_add(1);
        AccRange[0][0] = ID[0];
      });
    });
    myQueue.wait();
  }
  check("Counter = ", Counter, size * MagicY * MagicZ);
}

int main() {
  int x = 1500;
  try_1d_range<item<1>>(x);
  try_1d_range<id<1>>(x);
  try_2d_range<item<2>>(x);
  try_2d_range<id<2>>(x);
  try_3d_range<item<3>>(x);
  try_3d_range<id<3>>(x);
  try_unnamed_lambda(x);

  x = 256;
  try_1d_range<item<1>>(x);
  try_1d_range<id<1>>(x);
  try_2d_range<item<2>>(x);
  try_2d_range<id<2>>(x);
  try_3d_range<item<3>>(x);
  try_3d_range<id<3>>(x);
  try_unnamed_lambda(x);
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
