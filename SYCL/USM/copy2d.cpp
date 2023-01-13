//==---- copy2d.cpp - USM 2D copy test -------------------------------------==//
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

// Temporarily disabled until the failure is addressed.
// UNSUPPORTED: gpu-intel-pvc

#include <sycl/sycl.hpp>

#include "memops2d_utils.hpp"

using namespace sycl;

constexpr size_t RECT_WIDTH = 100;
constexpr size_t RECT_HEIGHT = 41;

template <typename T, OperationPath PathKind>
event doCopy2D(queue &Q, const T *Src, size_t SrcPitch, T *Dest,
               size_t DestPitch, size_t Width, size_t Height,
               std::vector<event> DepEvents) {
  if constexpr (PathKind == OperationPath::Expanded) {
    sycl::event::wait(DepEvents);
    return Q.submit([&](handler &CGH) {
      CGH.ext_oneapi_copy2d(Src, SrcPitch, Dest, DestPitch, Width, Height);
    });
  }
  if constexpr (PathKind == OperationPath::ExpandedDependsOn) {
    return Q.submit([&](handler &CGH) {
      CGH.depends_on(DepEvents);
      CGH.ext_oneapi_copy2d(Src, SrcPitch, Dest, DestPitch, Width, Height);
    });
  }
  if constexpr (PathKind == OperationPath::ShortcutNoEvent) {
    sycl::event::wait(DepEvents);
    return Q.ext_oneapi_copy2d(Src, SrcPitch, Dest, DestPitch, Width, Height);
  }
  if constexpr (PathKind == OperationPath::ShortcutOneEvent) {
    assert(DepEvents.size() && "No events in dependencies!");
    // wait on all other events than the first.
    for (size_t I = 1; I < DepEvents.size(); ++I)
      DepEvents[I].wait();
    return Q.ext_oneapi_copy2d(Src, SrcPitch, Dest, DestPitch, Width, Height,
                               DepEvents[0]);
  }
  if constexpr (PathKind == OperationPath::ShortcutEventList) {
    return Q.ext_oneapi_copy2d(Src, SrcPitch, Dest, DestPitch, Width, Height,
                               DepEvents);
  }
}

template <typename T, usm::alloc AllocKind, OperationPath PathKind>
int test(queue &Q, T ExpectedVal1, T ExpectedVal2) {
  int Failures = 0;

  // Test 1 - 2D copy entire buffer.
  {
    constexpr size_t SRC_ELEMS = RECT_WIDTH * RECT_HEIGHT;
    constexpr size_t DST_ELEMS = SRC_ELEMS;

    T *USMMemSrc = sycl::malloc<T>(SRC_ELEMS, Q, AllocKind);
    T *USMMemDst = sycl::malloc<T>(DST_ELEMS, Q, AllocKind);
    event SrcFillEvent = Q.fill(USMMemSrc, ExpectedVal1, SRC_ELEMS);
    event DstMemsetEvent = Q.memset(USMMemDst, 0, DST_ELEMS * sizeof(T));
    doCopy2D<T, PathKind>(Q, USMMemSrc, RECT_WIDTH, USMMemDst, RECT_WIDTH,
                          RECT_WIDTH, RECT_HEIGHT,
                          {SrcFillEvent, DstMemsetEvent})
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

    sycl::free(USMMemSrc, Q);
    sycl::free(USMMemDst, Q);
  }

  // Test 2 - 2D copy to vertically adjacent regions.
  {
    constexpr size_t SRC_ELEMS = RECT_WIDTH * RECT_HEIGHT;
    constexpr size_t DST_ELEMS = 2 * SRC_ELEMS;

    T *USMMemSrc1 = sycl::malloc<T>(SRC_ELEMS, Q, AllocKind);
    T *USMMemSrc2 = sycl::malloc<T>(SRC_ELEMS, Q, AllocKind);
    T *USMMemDst = sycl::malloc<T>(DST_ELEMS, Q, AllocKind);
    event Src1FillEvent = Q.fill(USMMemSrc1, ExpectedVal1, SRC_ELEMS);
    event Src2FillEvent = Q.fill(USMMemSrc2, ExpectedVal2, SRC_ELEMS);
    event DstMemsetEvent = Q.memset(USMMemDst, 0, DST_ELEMS * sizeof(T));
    event FirstCopyEvent = doCopy2D<T, PathKind>(
        Q, USMMemSrc1, RECT_WIDTH, USMMemDst, RECT_WIDTH, RECT_WIDTH,
        RECT_HEIGHT, {Src1FillEvent, Src2FillEvent, DstMemsetEvent});
    doCopy2D<T, PathKind>(
        Q, USMMemSrc2, RECT_WIDTH, USMMemDst + SRC_ELEMS, RECT_WIDTH,
        RECT_WIDTH, RECT_HEIGHT,
        {FirstCopyEvent, Src1FillEvent, Src2FillEvent, DstMemsetEvent})
        .wait();
    std::vector<T> Results;
    Results.resize(DST_ELEMS);
    Q.copy(USMMemDst, Results.data(), DST_ELEMS).wait();

    for (size_t I = 0; I < DST_ELEMS; ++I) {
      T ExpectedVal = I >= SRC_ELEMS ? ExpectedVal2 : ExpectedVal1;
      if (!checkResult<AllocKind, PathKind>(Results[I], ExpectedVal, I,
                                            "Test 2")) {
        ++Failures;
        break;
      }
    }

    sycl::free(USMMemSrc1, Q);
    sycl::free(USMMemSrc2, Q);
    sycl::free(USMMemDst, Q);
  }

  // Test 3 - 2D copy to horizontally adjacent regions.
  {
    constexpr size_t SRC_ELEMS = RECT_WIDTH * RECT_HEIGHT;
    constexpr size_t DST_ELEMS = 2 * SRC_ELEMS;

    T *USMMemSrc1 = sycl::malloc<T>(SRC_ELEMS, Q, AllocKind);
    T *USMMemSrc2 = sycl::malloc<T>(SRC_ELEMS, Q, AllocKind);
    T *USMMemDst = sycl::malloc<T>(DST_ELEMS, Q, AllocKind);
    event Src1FillEvent = Q.fill(USMMemSrc1, ExpectedVal1, SRC_ELEMS);
    event Src2FillEvent = Q.fill(USMMemSrc2, ExpectedVal2, SRC_ELEMS);
    event DstMemsetEvent = Q.memset(USMMemDst, 0, DST_ELEMS * sizeof(T));
    event FirstCopyEvent = doCopy2D<T, PathKind>(
        Q, USMMemSrc1, RECT_WIDTH, USMMemDst, 2 * RECT_WIDTH, RECT_WIDTH,
        RECT_HEIGHT, {Src1FillEvent, Src2FillEvent, DstMemsetEvent});
    doCopy2D<T, PathKind>(
        Q, USMMemSrc2, RECT_WIDTH, USMMemDst + RECT_WIDTH, 2 * RECT_WIDTH,
        RECT_WIDTH, RECT_HEIGHT,
        {FirstCopyEvent, Src1FillEvent, Src2FillEvent, DstMemsetEvent})
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

    sycl::free(USMMemSrc1, Q);
    sycl::free(USMMemSrc2, Q);
    sycl::free(USMMemDst, Q);
  }

  // Test 4 - 2D copy to 2x2 grid of rectangles.
  {
    constexpr size_t SRC_ELEMS = RECT_WIDTH * RECT_HEIGHT;
    constexpr size_t DST_ELEMS = 4 * SRC_ELEMS;

    T *USMMemSrc1 = sycl::malloc<T>(SRC_ELEMS, Q, AllocKind);
    T *USMMemSrc2 = sycl::malloc<T>(SRC_ELEMS, Q, AllocKind);
    T *USMMemDst = sycl::malloc<T>(DST_ELEMS, Q, AllocKind);
    event Src1FillEvent = Q.fill(USMMemSrc1, ExpectedVal1, SRC_ELEMS);
    event Src2FillEvent = Q.fill(USMMemSrc2, ExpectedVal2, SRC_ELEMS);
    event DstMemsetEvent = Q.memset(USMMemDst, 0, DST_ELEMS * sizeof(T));
    // Top left rectangle.
    event FirstCopyEvent = doCopy2D<T, PathKind>(
        Q, USMMemSrc1, RECT_WIDTH, USMMemDst, 2 * RECT_WIDTH, RECT_WIDTH,
        RECT_HEIGHT, {Src1FillEvent, Src2FillEvent, DstMemsetEvent});
    // Top right rectangle.
    event SecondCopyEvent = doCopy2D<T, PathKind>(
        Q, USMMemSrc2, RECT_WIDTH, USMMemDst + RECT_WIDTH, 2 * RECT_WIDTH,
        RECT_WIDTH, RECT_HEIGHT,
        {FirstCopyEvent, Src1FillEvent, Src2FillEvent, DstMemsetEvent});
    // Bottom left rectangle.
    event ThirdCopyEvent = doCopy2D<T, PathKind>(
        Q, USMMemSrc2, RECT_WIDTH, USMMemDst + 2 * SRC_ELEMS, 2 * RECT_WIDTH,
        RECT_WIDTH, RECT_HEIGHT,
        {FirstCopyEvent, SecondCopyEvent, Src1FillEvent, Src2FillEvent,
         DstMemsetEvent});
    // Bottom right rectangle.
    doCopy2D<T, PathKind>(Q, USMMemSrc1, RECT_WIDTH,
                          USMMemDst + 2 * SRC_ELEMS + RECT_WIDTH,
                          2 * RECT_WIDTH, RECT_WIDTH, RECT_HEIGHT,
                          {FirstCopyEvent, SecondCopyEvent, ThirdCopyEvent,
                           Src1FillEvent, Src2FillEvent, DstMemsetEvent})
        .wait();
    std::vector<T> Results;
    Results.resize(DST_ELEMS);
    Q.copy(USMMemDst, Results.data(), DST_ELEMS).wait();

    for (size_t I = 0; I < DST_ELEMS; ++I) {
      T ExpectedVal = ((I / RECT_WIDTH) + (I / (2 * SRC_ELEMS))) % 2
                          ? ExpectedVal2
                          : ExpectedVal1;
      if (!checkResult<AllocKind, PathKind>(Results[I], ExpectedVal, I,
                                            "Test 4")) {
        ++Failures;
        break;
      }
    }

    sycl::free(USMMemSrc1, Q);
    sycl::free(USMMemSrc2, Q);
    sycl::free(USMMemDst, Q);
  }

  // Test 5 - 2D copy from vertically adjacent regions.
  {
    constexpr size_t SRC_ELEMS = 2 * RECT_WIDTH * RECT_HEIGHT;
    constexpr size_t DST_ELEMS = SRC_ELEMS / 2;

    T *USMMemSrc = sycl::malloc<T>(SRC_ELEMS, Q, AllocKind);
    T *USMMemDst1 = sycl::malloc<T>(DST_ELEMS, Q, AllocKind);
    T *USMMemDst2 = sycl::malloc<T>(DST_ELEMS, Q, AllocKind);
    event Src1FillEvent = Q.fill(USMMemSrc, ExpectedVal1, DST_ELEMS);
    event Src2FillEvent =
        Q.fill(USMMemSrc + DST_ELEMS, ExpectedVal2, DST_ELEMS);
    event Dst1MemsetEvent = Q.memset(USMMemDst1, 0, DST_ELEMS * sizeof(T));
    event Dst2MemsetEvent = Q.memset(USMMemDst2, 0, DST_ELEMS * sizeof(T));
    event FirstCopyEvent = doCopy2D<T, PathKind>(
        Q, USMMemSrc, RECT_WIDTH, USMMemDst1, RECT_WIDTH, RECT_WIDTH,
        RECT_HEIGHT,
        {Src1FillEvent, Src2FillEvent, Dst1MemsetEvent, Dst2MemsetEvent});
    doCopy2D<T, PathKind>(Q, USMMemSrc + DST_ELEMS, RECT_WIDTH, USMMemDst2,
                          RECT_WIDTH, RECT_WIDTH, RECT_HEIGHT,
                          {FirstCopyEvent, Src1FillEvent, Src2FillEvent,
                           Dst1MemsetEvent, Dst2MemsetEvent})
        .wait();
    std::vector<T> Results;
    Results.resize(SRC_ELEMS);
    Q.copy(USMMemDst1, Results.data(), DST_ELEMS);
    Q.copy(USMMemDst2, Results.data() + DST_ELEMS, DST_ELEMS);
    Q.wait();

    for (size_t I = 0; I < SRC_ELEMS; ++I) {
      T ExpectedVal = I >= DST_ELEMS ? ExpectedVal2 : ExpectedVal1;
      if (!checkResult<AllocKind, PathKind>(Results[I], ExpectedVal, I,
                                            "Test 5")) {
        ++Failures;
        break;
      }
    }

    sycl::free(USMMemSrc, Q);
    sycl::free(USMMemDst1, Q);
    sycl::free(USMMemDst2, Q);
  }

  // Test 6 - 2D copy from horizontally adjacent regions.
  {
    constexpr size_t SRC_ELEMS = 2 * RECT_WIDTH * RECT_HEIGHT;
    constexpr size_t DST_ELEMS = SRC_ELEMS / 2;

    T *USMMemSrc = sycl::malloc<T>(SRC_ELEMS, Q, AllocKind);
    T *USMMemDst1 = sycl::malloc<T>(DST_ELEMS, Q, AllocKind);
    T *USMMemDst2 = sycl::malloc<T>(DST_ELEMS, Q, AllocKind);
    event SrcFillEvent = Q.parallel_for(SRC_ELEMS, [=](item<1> Id) {
      USMMemSrc[Id] = (Id / RECT_WIDTH) % 2 ? ExpectedVal2 : ExpectedVal1;
    });
    event Dst1MemsetEvent = Q.memset(USMMemDst1, 0, DST_ELEMS * sizeof(T));
    event Dst2MemsetEvent = Q.memset(USMMemDst2, 0, DST_ELEMS * sizeof(T));
    event FirstCopyEvent = doCopy2D<T, PathKind>(
        Q, USMMemSrc, 2 * RECT_WIDTH, USMMemDst1, RECT_WIDTH, RECT_WIDTH,
        RECT_HEIGHT, {SrcFillEvent, Dst1MemsetEvent, Dst2MemsetEvent});
    doCopy2D<T, PathKind>(
        Q, USMMemSrc + RECT_WIDTH, 2 * RECT_WIDTH, USMMemDst2, RECT_WIDTH,
        RECT_WIDTH, RECT_HEIGHT,
        {FirstCopyEvent, SrcFillEvent, Dst1MemsetEvent, Dst2MemsetEvent})
        .wait();
    std::vector<T> Results;
    Results.resize(SRC_ELEMS);
    Q.copy(USMMemDst1, Results.data(), DST_ELEMS);
    Q.copy(USMMemDst2, Results.data() + DST_ELEMS, DST_ELEMS);
    Q.wait();

    for (size_t I = 0; I < SRC_ELEMS; ++I) {
      T ExpectedVal = I >= DST_ELEMS ? ExpectedVal2 : ExpectedVal1;
      if (!checkResult<AllocKind, PathKind>(Results[I], ExpectedVal, I,
                                            "Test 6")) {
        ++Failures;
        break;
      }
    }

    sycl::free(USMMemSrc, Q);
    sycl::free(USMMemDst1, Q);
    sycl::free(USMMemDst2, Q);
  }

  // Test 7 - 2D copy from 2x2 grid of rectangles.
  {
    constexpr size_t SRC_ELEMS = 4 * RECT_WIDTH * RECT_HEIGHT;
    constexpr size_t DST_ELEMS = SRC_ELEMS / 4;

    T *USMMemSrc = sycl::malloc<T>(SRC_ELEMS, Q, AllocKind);
    T *USMMemDst1 = sycl::malloc<T>(DST_ELEMS, Q, AllocKind);
    T *USMMemDst2 = sycl::malloc<T>(DST_ELEMS, Q, AllocKind);
    T *USMMemDst3 = sycl::malloc<T>(DST_ELEMS, Q, AllocKind);
    T *USMMemDst4 = sycl::malloc<T>(DST_ELEMS, Q, AllocKind);
    event SrcFillEvent = Q.parallel_for(SRC_ELEMS, [=](item<1> Id) {
      USMMemSrc[Id] = ((Id / RECT_WIDTH) + (Id / (2 * DST_ELEMS))) % 2
                          ? ExpectedVal2
                          : ExpectedVal1;
    });
    event Dst1MemsetEvent = Q.memset(USMMemDst1, 0, DST_ELEMS * sizeof(T));
    event Dst2MemsetEvent = Q.memset(USMMemDst2, 0, DST_ELEMS * sizeof(T));
    event Dst3MemsetEvent = Q.memset(USMMemDst3, 0, DST_ELEMS * sizeof(T));
    event Dst4MemsetEvent = Q.memset(USMMemDst4, 0, DST_ELEMS * sizeof(T));
    // Top left rectangle.
    event FirstCopyEvent =
        doCopy2D<T, PathKind>(Q, USMMemSrc, 2 * RECT_WIDTH, USMMemDst1,
                              RECT_WIDTH, RECT_WIDTH, RECT_HEIGHT,
                              {SrcFillEvent, Dst1MemsetEvent, Dst2MemsetEvent,
                               Dst3MemsetEvent, Dst4MemsetEvent});
    // Bottom right rectangle.
    event SecondCopyEvent = doCopy2D<T, PathKind>(
        Q, USMMemSrc + 2 * DST_ELEMS + RECT_WIDTH, 2 * RECT_WIDTH, USMMemDst2,
        RECT_WIDTH, RECT_WIDTH, RECT_HEIGHT,
        {FirstCopyEvent, SrcFillEvent, Dst1MemsetEvent, Dst2MemsetEvent,
         Dst3MemsetEvent, Dst4MemsetEvent});
    // Bottom left rectangle.
    event ThirdCopyEvent = doCopy2D<T, PathKind>(
        Q, USMMemSrc + 2 * DST_ELEMS, 2 * RECT_WIDTH, USMMemDst3, RECT_WIDTH,
        RECT_WIDTH, RECT_HEIGHT,
        {FirstCopyEvent, SecondCopyEvent, SrcFillEvent, Dst1MemsetEvent,
         Dst2MemsetEvent, Dst3MemsetEvent, Dst4MemsetEvent});
    // Top right rectangle.
    doCopy2D<T, PathKind>(Q, USMMemSrc + RECT_WIDTH, 2 * RECT_WIDTH, USMMemDst4,
                          RECT_WIDTH, RECT_WIDTH, RECT_HEIGHT,
                          {FirstCopyEvent, SecondCopyEvent, ThirdCopyEvent,
                           SrcFillEvent, Dst1MemsetEvent, Dst2MemsetEvent,
                           Dst3MemsetEvent, Dst4MemsetEvent})
        .wait();
    std::vector<T> Results;
    Results.resize(SRC_ELEMS);
    Q.copy(USMMemDst1, Results.data(), DST_ELEMS);
    Q.copy(USMMemDst2, Results.data() + DST_ELEMS, DST_ELEMS);
    Q.copy(USMMemDst3, Results.data() + 2 * DST_ELEMS, DST_ELEMS);
    Q.copy(USMMemDst4, Results.data() + 3 * DST_ELEMS, DST_ELEMS);
    Q.wait();

    for (size_t I = 0; I < SRC_ELEMS; ++I) {
      T ExpectedVal = I >= 2 * DST_ELEMS ? ExpectedVal2 : ExpectedVal1;
      if (!checkResult<AllocKind, PathKind>(Results[I], ExpectedVal, I,
                                            "Test 7")) {
        ++Failures;
        break;
      }
    }

    sycl::free(USMMemSrc, Q);
    sycl::free(USMMemDst1, Q);
    sycl::free(USMMemDst2, Q);
    sycl::free(USMMemDst3, Q);
    sycl::free(USMMemDst4, Q);
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
