//==---- fill2d.cpp - USM 2D fill test -------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RUN: %clangxx -fsycl -fsycl-device-code-split=per_kernel -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

#include "memops2d_utils.hpp"

using namespace sycl;

constexpr size_t RECT_WIDTH = 100;
constexpr size_t RECT_HEIGHT = 41;

template <typename T, OperationPath PathKind>
event doFill2D(queue &Q, void *Dest, size_t DestPitch, const T &Pattern,
               size_t Width, size_t Height, std::vector<event> DepEvents) {
  if constexpr (PathKind == OperationPath::Expanded) {
    sycl::event::wait(DepEvents);
    return Q.submit([&](handler &CGH) {
      CGH.ext_oneapi_fill2d(Dest, DestPitch, Pattern, Width, Height);
    });
  }
  if constexpr (PathKind == OperationPath::ExpandedDependsOn) {
    return Q.submit([&](handler &CGH) {
      CGH.depends_on(DepEvents);
      CGH.ext_oneapi_fill2d(Dest, DestPitch, Pattern, Width, Height);
    });
  }
  if constexpr (PathKind == OperationPath::ShortcutNoEvent) {
    sycl::event::wait(DepEvents);
    return Q.ext_oneapi_fill2d(Dest, DestPitch, Pattern, Width, Height);
  }
  if constexpr (PathKind == OperationPath::ShortcutOneEvent) {
    assert(DepEvents.size() && "No events in dependencies!");
    // wait on all other events than the first.
    for (size_t I = 1; I < DepEvents.size(); ++I)
      DepEvents[I].wait();
    return Q.ext_oneapi_fill2d(Dest, DestPitch, Pattern, Width, Height,
                               DepEvents[0]);
  }
  if constexpr (PathKind == OperationPath::ShortcutEventList) {
    return Q.ext_oneapi_fill2d(Dest, DestPitch, Pattern, Width, Height,
                               DepEvents);
  }
}

template <typename T, usm::alloc AllocKind, OperationPath PathKind>
int test(queue &Q, T ExpectedVal1, T ExpectedVal2) {
  int Failures = 0;

  // Test 1 - 2D fill entire buffer.
  {
    constexpr size_t DST_ELEMS = RECT_WIDTH * RECT_HEIGHT;

    T *USMMemDst = sycl::malloc<T>(DST_ELEMS, Q, AllocKind);
    event DstMemsetEvent = Q.memset(USMMemDst, 0, DST_ELEMS * sizeof(T));
    doFill2D<T, PathKind>(Q, USMMemDst, RECT_WIDTH, ExpectedVal1, RECT_WIDTH,
                          RECT_HEIGHT, {DstMemsetEvent})
        .wait();
    std::vector<T> Results;
    Results.resize(DST_ELEMS);
    Q.copy(USMMemDst, Results.data(), DST_ELEMS).wait();

    for (size_t I = 0; I < DST_ELEMS; ++I) {
      if (!checkResult<AllocKind, PathKind>(Results[I], ExpectedVal1, I,
                                            "Test 1")) {
        ++Failures;
        break;
      }
    }

    sycl::free(USMMemDst, Q);
  }

  // Test 2 - 2D fill vertically adjacent regions.
  {
    constexpr size_t DST_ELEMS = 2 * RECT_WIDTH * RECT_HEIGHT;

    T *USMMemDst = sycl::malloc<T>(DST_ELEMS, Q, AllocKind);
    event DstMemsetEvent = Q.memset(USMMemDst, 0, DST_ELEMS * sizeof(T));
    event FirstFillEvent =
        doFill2D<T, PathKind>(Q, USMMemDst, RECT_WIDTH, ExpectedVal1,
                              RECT_WIDTH, RECT_HEIGHT, {DstMemsetEvent});
    doFill2D<T, PathKind>(Q, USMMemDst + DST_ELEMS / 2, RECT_WIDTH,
                          ExpectedVal2, RECT_WIDTH, RECT_HEIGHT,
                          {FirstFillEvent, DstMemsetEvent})
        .wait();
    std::vector<T> Results;
    Results.resize(DST_ELEMS);
    Q.copy(USMMemDst, Results.data(), DST_ELEMS).wait();

    for (size_t I = 0; I < DST_ELEMS; ++I) {
      T ExpectedVal = I >= (DST_ELEMS / 2) ? ExpectedVal2 : ExpectedVal1;
      if (!checkResult<AllocKind, PathKind>(Results[I], ExpectedVal, I,
                                            "Test 2")) {
        ++Failures;
        break;
      }
    }

    sycl::free(USMMemDst, Q);
  }

  // Test 3 - 2D fill horizontally adjacent regions.
  {
    constexpr size_t DST_ELEMS = 2 * RECT_WIDTH * RECT_HEIGHT;

    T *USMMemDst = sycl::malloc<T>(DST_ELEMS, Q, AllocKind);
    event DstMemsetEvent = Q.memset(USMMemDst, 0, DST_ELEMS * sizeof(T));
    event FirstFillEvent =
        doFill2D<T, PathKind>(Q, USMMemDst, 2 * RECT_WIDTH, ExpectedVal1,
                              RECT_WIDTH, RECT_HEIGHT, {DstMemsetEvent});
    doFill2D<T, PathKind>(Q, USMMemDst + RECT_WIDTH, 2 * RECT_WIDTH,
                          ExpectedVal2, RECT_WIDTH, RECT_HEIGHT,
                          {FirstFillEvent, DstMemsetEvent})
        .wait();
    std::vector<T> Results;
    Results.resize(DST_ELEMS);
    Q.copy(USMMemDst, Results.data(), DST_ELEMS).wait();

    for (size_t I = 0; I < DST_ELEMS; ++I) {
      T ExpectedVal = (I / RECT_WIDTH) % 2 ? ExpectedVal2 : ExpectedVal1;
      if (!checkResult<AllocKind, PathKind>(Results[I], ExpectedVal, I,
                                            "Test 3")) {
        ++Failures;
        break;
      }
    }

    sycl::free(USMMemDst, Q);
  }

  // Test 4 - 2D fill 2x2 grid of rectangles.
  {
    constexpr size_t DST_ELEMS = 4 * RECT_WIDTH * RECT_HEIGHT;

    T *USMMemDst = sycl::malloc<T>(DST_ELEMS, Q, AllocKind);
    event DstMemsetEvent = Q.memset(USMMemDst, 0, DST_ELEMS * sizeof(T));
    // Top left rectangle.
    event FirstFillEvent =
        doFill2D<T, PathKind>(Q, USMMemDst, 2 * RECT_WIDTH, ExpectedVal1,
                              RECT_WIDTH, RECT_HEIGHT, {DstMemsetEvent});
    // Top right rectangle.
    event SecondFillEvent = doFill2D<T, PathKind>(
        Q, USMMemDst + RECT_WIDTH, 2 * RECT_WIDTH, ExpectedVal2, RECT_WIDTH,
        RECT_HEIGHT, {FirstFillEvent, DstMemsetEvent});
    // Bottom left rectangle.
    event ThirdFillEvent = doFill2D<T, PathKind>(
        Q, USMMemDst + DST_ELEMS / 2, 2 * RECT_WIDTH, ExpectedVal2, RECT_WIDTH,
        RECT_HEIGHT, {FirstFillEvent, SecondFillEvent, DstMemsetEvent});
    // Bottom right rectangle.
    doFill2D<T, PathKind>(
        Q, USMMemDst + DST_ELEMS / 2 + RECT_WIDTH, 2 * RECT_WIDTH, ExpectedVal1,
        RECT_WIDTH, RECT_HEIGHT,
        {FirstFillEvent, SecondFillEvent, ThirdFillEvent, DstMemsetEvent})
        .wait();
    std::vector<T> Results;
    Results.resize(DST_ELEMS);
    Q.copy(USMMemDst, Results.data(), DST_ELEMS).wait();

    for (size_t I = 0; I < DST_ELEMS; ++I) {
      T ExpectedVal = ((I / RECT_WIDTH) + (I / (DST_ELEMS / 2))) % 2
                          ? ExpectedVal2
                          : ExpectedVal1;
      if (!checkResult<AllocKind, PathKind>(Results[I], ExpectedVal, I,
                                            "Test 4")) {
        ++Failures;
        break;
      }
    }

    sycl::free(USMMemDst, Q);
  }

  return Failures;
}

template <typename T, usm::alloc AllocKind>
int testForAllPaths(queue &Q, T ExpectedVal1, T ExpectedVal2) {
  int Failures = 0;
  Failures += test<T, AllocKind, OperationPath::Expanded>(Q, ExpectedVal1,
                                                          ExpectedVal2);
  Failures += test<T, AllocKind, OperationPath::ExpandedDependsOn>(
      Q, ExpectedVal1, ExpectedVal2);
  Failures += test<T, AllocKind, OperationPath::ShortcutNoEvent>(
      Q, ExpectedVal1, ExpectedVal2);
  Failures += test<T, AllocKind, OperationPath::ShortcutOneEvent>(
      Q, ExpectedVal1, ExpectedVal2);
  Failures += test<T, AllocKind, OperationPath::ShortcutEventList>(
      Q, ExpectedVal1, ExpectedVal2);
  return Failures;
}

template <usm::alloc AllocKind> int testForAllTypesAndPaths(queue &Q) {

  bool SupportsHalf = Q.get_device().has(aspect::fp16);
  bool SupportsDouble = Q.get_device().has(aspect::fp64);

  TestStruct TestStructRef1{42, 'f'}, TestStructRef2{1234, 'd'};

  int Failures = 0;
  Failures += testForAllPaths<char, AllocKind>(Q, 'f', 'd');
  Failures += testForAllPaths<short, AllocKind>(Q, 1234, 42);
  Failures += testForAllPaths<int, AllocKind>(Q, 42, 1234);
  Failures += testForAllPaths<long, AllocKind>(Q, 1242, 34);
  Failures += testForAllPaths<long long, AllocKind>(Q, 34, 1242);
  if (SupportsHalf)
    Failures += testForAllPaths<sycl::half, AllocKind>(Q, 12.34f, 42.24f);
  Failures += testForAllPaths<float, AllocKind>(Q, 42.24f, 12.34f);
  if (SupportsDouble)
    Failures += testForAllPaths<double, AllocKind>(Q, 42.34, 12.24);
  Failures +=
      testForAllPaths<TestStruct, AllocKind>(Q, TestStructRef1, TestStructRef2);
  return Failures;
}

int main() {
  queue Q;

  int Failures = 0;
  if (Q.get_device().has(aspect::usm_device_allocations))
    Failures += testForAllTypesAndPaths<usm::alloc::device>(Q);
  if (Q.get_device().has(aspect::usm_host_allocations))
    Failures += testForAllTypesAndPaths<usm::alloc::host>(Q);
  if (Q.get_device().has(aspect::usm_shared_allocations))
    Failures += testForAllTypesAndPaths<usm::alloc::shared>(Q);

  if (!Failures)
    std::cout << "Passed!" << std::endl;

  return Failures;
}
