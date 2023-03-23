//==---- memcpy2d.cpp - USM 2D memcpy test ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// Temporarily disabled until the failure is addressed.
// UNSUPPORTED: gpu-intel-pvc

#include <sycl/sycl.hpp>

#include "memops2d_utils.hpp"

using namespace sycl;

constexpr size_t RECT_WIDTH = 30;
constexpr size_t RECT_HEIGHT = 21;

template <typename T, OperationPath PathKind>
event doMemcpy2D(queue &Q, void *Dest, size_t DestPitch, const void *Src,
                 size_t SrcPitch, size_t Width, size_t Height,
                 std::vector<event> DepEvents) {
  if constexpr (PathKind == OperationPath::Expanded) {
    sycl::event::wait(DepEvents);
    return Q.submit([&](handler &CGH) {
      CGH.ext_oneapi_memcpy2d(Dest, DestPitch, Src, SrcPitch, Width, Height);
    });
  }
  if constexpr (PathKind == OperationPath::ExpandedDependsOn) {
    return Q.submit([&](handler &CGH) {
      CGH.depends_on(DepEvents);
      CGH.ext_oneapi_memcpy2d(Dest, DestPitch, Src, SrcPitch, Width, Height);
    });
  }
  if constexpr (PathKind == OperationPath::ShortcutNoEvent) {
    sycl::event::wait(DepEvents);
    return Q.ext_oneapi_memcpy2d(Dest, DestPitch, Src, SrcPitch, Width, Height);
  }
  if constexpr (PathKind == OperationPath::ShortcutOneEvent) {
    assert(DepEvents.size() && "No events in dependencies!");
    // wait on all other events than the first.
    for (size_t I = 1; I < DepEvents.size(); ++I)
      DepEvents[I].wait();
    return Q.ext_oneapi_memcpy2d(Dest, DestPitch, Src, SrcPitch, Width, Height,
                                 DepEvents[0]);
  }
  if constexpr (PathKind == OperationPath::ShortcutEventList) {
    return Q.ext_oneapi_memcpy2d(Dest, DestPitch, Src, SrcPitch, Width, Height,
                                 DepEvents);
  }
}

template <typename T, Alloc SrcAllocKind, Alloc DstAllocKind,
          OperationPath PathKind>
int test(queue &Q, T ExpectedVal1, T ExpectedVal2) {
  int Failures = 0;

  // Test 1 - 2D copy entire buffer.
  {
    constexpr size_t SRC_ELEMS = RECT_WIDTH * RECT_HEIGHT;
    constexpr size_t DST_ELEMS = SRC_ELEMS;

    T *USMMemSrc = allocate<T, SrcAllocKind>(SRC_ELEMS, Q);
    T *USMMemDst = allocate<T, DstAllocKind>(DST_ELEMS, Q);
    event SrcFillEvent =
        fill<SrcAllocKind>(Q, USMMemSrc, ExpectedVal1, SRC_ELEMS);
    event DstMemsetEvent =
        memset<DstAllocKind>(Q, USMMemDst, 0, DST_ELEMS * sizeof(T));
    doMemcpy2D<T, PathKind>(Q, USMMemDst, RECT_WIDTH * sizeof(T), USMMemSrc,
                            RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T),
                            RECT_HEIGHT, {SrcFillEvent, DstMemsetEvent})
        .wait();
    std::vector<T> Results;
    Results.resize(DST_ELEMS);
    copy_to_host<DstAllocKind>(Q, USMMemDst, Results.data(), DST_ELEMS).wait();

    for (size_t I = 0; I < DST_ELEMS; ++I) {
      if (!checkResult<SrcAllocKind, DstAllocKind, PathKind>(
              Results[I], ExpectedVal1, I, "Test 1")) {
        ++Failures;
        break;
      }
    }

    free<SrcAllocKind>(USMMemSrc, Q);
    free<DstAllocKind>(USMMemDst, Q);
  }

  // Test 2 - 2D copy to vertically adjacent regions.
  {
    constexpr size_t SRC_ELEMS = RECT_WIDTH * RECT_HEIGHT;
    constexpr size_t DST_ELEMS = 2 * SRC_ELEMS;

    T *USMMemSrc1 = allocate<T, SrcAllocKind>(SRC_ELEMS, Q);
    T *USMMemSrc2 = allocate<T, SrcAllocKind>(SRC_ELEMS, Q);
    T *USMMemDst = allocate<T, DstAllocKind>(DST_ELEMS, Q);
    event Src1FillEvent =
        fill<SrcAllocKind>(Q, USMMemSrc1, ExpectedVal1, SRC_ELEMS);
    event Src2FillEvent =
        fill<SrcAllocKind>(Q, USMMemSrc2, ExpectedVal2, SRC_ELEMS);
    event DstMemsetEvent =
        memset<DstAllocKind>(Q, USMMemDst, 0, DST_ELEMS * sizeof(T));
    event FirstMemcpyEvent = doMemcpy2D<T, PathKind>(
        Q, USMMemDst, RECT_WIDTH * sizeof(T), USMMemSrc1,
        RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT,
        {Src1FillEvent, Src2FillEvent, DstMemsetEvent});
    doMemcpy2D<T, PathKind>(
        Q, USMMemDst + SRC_ELEMS, RECT_WIDTH * sizeof(T), USMMemSrc2,
        RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT,
        {FirstMemcpyEvent, Src1FillEvent, Src2FillEvent, DstMemsetEvent})
        .wait();
    std::vector<T> Results;
    Results.resize(DST_ELEMS);
    copy_to_host<DstAllocKind>(Q, USMMemDst, Results.data(), DST_ELEMS).wait();

    for (size_t I = 0; I < DST_ELEMS; ++I) {
      T ExpectedVal = I >= SRC_ELEMS ? ExpectedVal2 : ExpectedVal1;
      if (!checkResult<SrcAllocKind, DstAllocKind, PathKind>(
              Results[I], ExpectedVal, I, "Test 2")) {
        ++Failures;
        break;
      }
    }

    free<SrcAllocKind>(USMMemSrc1, Q);
    free<SrcAllocKind>(USMMemSrc2, Q);
    free<DstAllocKind>(USMMemDst, Q);
  }

  // Test 3 - 2D copy to horizontally adjacent regions.
  {
    constexpr size_t SRC_ELEMS = RECT_WIDTH * RECT_HEIGHT;
    constexpr size_t DST_ELEMS = 2 * SRC_ELEMS;

    T *USMMemSrc1 = allocate<T, SrcAllocKind>(SRC_ELEMS, Q);
    T *USMMemSrc2 = allocate<T, SrcAllocKind>(SRC_ELEMS, Q);
    T *USMMemDst = allocate<T, DstAllocKind>(DST_ELEMS, Q);
    event Src1FillEvent =
        fill<SrcAllocKind>(Q, USMMemSrc1, ExpectedVal1, SRC_ELEMS);
    event Src2FillEvent =
        fill<SrcAllocKind>(Q, USMMemSrc2, ExpectedVal2, SRC_ELEMS);
    event DstMemsetEvent =
        memset<DstAllocKind>(Q, USMMemDst, 0, DST_ELEMS * sizeof(T));
    event FirstMemcpyEvent = doMemcpy2D<T, PathKind>(
        Q, USMMemDst, 2 * RECT_WIDTH * sizeof(T), USMMemSrc1,
        RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT,
        {Src1FillEvent, Src2FillEvent, DstMemsetEvent});
    doMemcpy2D<T, PathKind>(
        Q, USMMemDst + RECT_WIDTH, 2 * RECT_WIDTH * sizeof(T), USMMemSrc2,
        RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT,
        {FirstMemcpyEvent, Src1FillEvent, Src2FillEvent, DstMemsetEvent})
        .wait();
    std::vector<T> Results;
    Results.resize(DST_ELEMS);
    copy_to_host<DstAllocKind>(Q, USMMemDst, Results.data(), DST_ELEMS).wait();

    for (size_t I = 0; I < DST_ELEMS; ++I) {
      T ExpectedVal = (I / RECT_WIDTH) % 2 ? ExpectedVal2 : ExpectedVal1;
      if (!checkResult<SrcAllocKind, DstAllocKind, PathKind>(
              Results[I], ExpectedVal, I, "Test 3")) {
        ++Failures;
        break;
      }
    }

    free<SrcAllocKind>(USMMemSrc1, Q);
    free<SrcAllocKind>(USMMemSrc2, Q);
    free<DstAllocKind>(USMMemDst, Q);
  }

  // Test 4 - 2D copy to 2x2 grid of rectangles.
  {
    constexpr size_t SRC_ELEMS = RECT_WIDTH * RECT_HEIGHT;
    constexpr size_t DST_ELEMS = 4 * SRC_ELEMS;

    T *USMMemSrc1 = allocate<T, SrcAllocKind>(SRC_ELEMS, Q);
    T *USMMemSrc2 = allocate<T, SrcAllocKind>(SRC_ELEMS, Q);
    T *USMMemDst = allocate<T, DstAllocKind>(DST_ELEMS, Q);
    event Src1FillEvent =
        fill<SrcAllocKind>(Q, USMMemSrc1, ExpectedVal1, SRC_ELEMS);
    event Src2FillEvent =
        fill<SrcAllocKind>(Q, USMMemSrc2, ExpectedVal2, SRC_ELEMS);
    event DstMemsetEvent =
        memset<DstAllocKind>(Q, USMMemDst, 0, DST_ELEMS * sizeof(T));
    // Top left rectangle.
    event FirstMemcpyEvent = doMemcpy2D<T, PathKind>(
        Q, USMMemDst, 2 * RECT_WIDTH * sizeof(T), USMMemSrc1,
        RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT,
        {Src1FillEvent, Src2FillEvent, DstMemsetEvent});
    // Top right rectangle.
    event SecondMemcpyEvent = doMemcpy2D<T, PathKind>(
        Q, USMMemDst + RECT_WIDTH, 2 * RECT_WIDTH * sizeof(T), USMMemSrc2,
        RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT,
        {FirstMemcpyEvent, Src1FillEvent, Src2FillEvent, DstMemsetEvent});
    // Bottom left rectangle.
    event ThirdMemcpyEvent = doMemcpy2D<T, PathKind>(
        Q, USMMemDst + 2 * SRC_ELEMS, 2 * RECT_WIDTH * sizeof(T), USMMemSrc2,
        RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT,
        {FirstMemcpyEvent, SecondMemcpyEvent, Src1FillEvent, Src2FillEvent,
         DstMemsetEvent});
    // Bottom right rectangle.
    doMemcpy2D<T, PathKind>(
        Q, USMMemDst + 2 * SRC_ELEMS + RECT_WIDTH, 2 * RECT_WIDTH * sizeof(T),
        USMMemSrc1, RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT,
        {FirstMemcpyEvent, SecondMemcpyEvent, ThirdMemcpyEvent, Src1FillEvent,
         Src2FillEvent, DstMemsetEvent})
        .wait();
    std::vector<T> Results;
    Results.resize(DST_ELEMS);
    copy_to_host<DstAllocKind>(Q, USMMemDst, Results.data(), DST_ELEMS).wait();

    for (size_t I = 0; I < DST_ELEMS; ++I) {
      T ExpectedVal = ((I / RECT_WIDTH) + (I / (2 * SRC_ELEMS))) % 2
                          ? ExpectedVal2
                          : ExpectedVal1;
      if (!checkResult<SrcAllocKind, DstAllocKind, PathKind>(
              Results[I], ExpectedVal, I, "Test 4")) {
        ++Failures;
        break;
      }
    }

    free<SrcAllocKind>(USMMemSrc1, Q);
    free<SrcAllocKind>(USMMemSrc2, Q);
    free<DstAllocKind>(USMMemDst, Q);
  }

  // Test 5 - 2D copy from vertically adjacent regions.
  {
    constexpr size_t SRC_ELEMS = 2 * RECT_WIDTH * RECT_HEIGHT;
    constexpr size_t DST_ELEMS = SRC_ELEMS / 2;

    T *USMMemSrc = allocate<T, SrcAllocKind>(SRC_ELEMS, Q);
    T *USMMemDst1 = allocate<T, DstAllocKind>(DST_ELEMS, Q);
    T *USMMemDst2 = allocate<T, DstAllocKind>(DST_ELEMS, Q);
    event SrcFillEvent =
        fill_with<SrcAllocKind>(Q, USMMemSrc, SRC_ELEMS, [=](size_t I) {
          return I < DST_ELEMS ? ExpectedVal1 : ExpectedVal2;
        });
    event Dst1MemsetEvent =
        memset<DstAllocKind>(Q, USMMemDst1, 0, DST_ELEMS * sizeof(T));
    event Dst2MemsetEvent =
        memset<DstAllocKind>(Q, USMMemDst2, 0, DST_ELEMS * sizeof(T));
    event FirstMemcpyEvent = doMemcpy2D<T, PathKind>(
        Q, USMMemDst1, RECT_WIDTH * sizeof(T), USMMemSrc,
        RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT,
        {SrcFillEvent, Dst1MemsetEvent, Dst2MemsetEvent});
    doMemcpy2D<T, PathKind>(
        Q, USMMemDst2, RECT_WIDTH * sizeof(T), USMMemSrc + DST_ELEMS,
        RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT,
        {FirstMemcpyEvent, SrcFillEvent, Dst1MemsetEvent, Dst2MemsetEvent})
        .wait();
    std::vector<T> Results;
    Results.resize(SRC_ELEMS);
    copy_to_host<DstAllocKind>(Q, USMMemDst1, Results.data(), DST_ELEMS);
    copy_to_host<DstAllocKind>(Q, USMMemDst2, Results.data() + DST_ELEMS,
                               DST_ELEMS);
    Q.wait();

    for (size_t I = 0; I < SRC_ELEMS; ++I) {
      T ExpectedVal = I >= DST_ELEMS ? ExpectedVal2 : ExpectedVal1;
      if (!checkResult<SrcAllocKind, DstAllocKind, PathKind>(
              Results[I], ExpectedVal, I, "Test 5")) {
        ++Failures;
        break;
      }
    }

    free<SrcAllocKind>(USMMemSrc, Q);
    free<DstAllocKind>(USMMemDst1, Q);
    free<DstAllocKind>(USMMemDst2, Q);
  }

  // Test 6 - 2D copy from horizontally adjacent regions.
  {
    constexpr size_t SRC_ELEMS = 2 * RECT_WIDTH * RECT_HEIGHT;
    constexpr size_t DST_ELEMS = SRC_ELEMS / 2;

    T *USMMemSrc = allocate<T, SrcAllocKind>(SRC_ELEMS, Q);
    T *USMMemDst1 = allocate<T, DstAllocKind>(DST_ELEMS, Q);
    T *USMMemDst2 = allocate<T, DstAllocKind>(DST_ELEMS, Q);
    event SrcFillEvent =
        fill_with<SrcAllocKind>(Q, USMMemSrc, SRC_ELEMS, [=](size_t I) {
          return (I / RECT_WIDTH) % 2 ? ExpectedVal2 : ExpectedVal1;
        });
    event Dst1MemsetEvent =
        memset<DstAllocKind>(Q, USMMemDst1, 0, DST_ELEMS * sizeof(T));
    event Dst2MemsetEvent =
        memset<DstAllocKind>(Q, USMMemDst2, 0, DST_ELEMS * sizeof(T));
    event FirstMemcpyEvent = doMemcpy2D<T, PathKind>(
        Q, USMMemDst1, RECT_WIDTH * sizeof(T), USMMemSrc,
        2 * RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT,
        {SrcFillEvent, Dst1MemsetEvent, Dst2MemsetEvent});
    doMemcpy2D<T, PathKind>(
        Q, USMMemDst2, RECT_WIDTH * sizeof(T), USMMemSrc + RECT_WIDTH,
        2 * RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT,
        {FirstMemcpyEvent, SrcFillEvent, Dst1MemsetEvent, Dst2MemsetEvent})
        .wait();
    std::vector<T> Results;
    Results.resize(SRC_ELEMS);
    copy_to_host<DstAllocKind>(Q, USMMemDst1, Results.data(), DST_ELEMS);
    copy_to_host<DstAllocKind>(Q, USMMemDst2, Results.data() + DST_ELEMS,
                               DST_ELEMS);
    Q.wait();

    for (size_t I = 0; I < SRC_ELEMS; ++I) {
      T ExpectedVal = I >= DST_ELEMS ? ExpectedVal2 : ExpectedVal1;
      if (!checkResult<SrcAllocKind, DstAllocKind, PathKind>(
              Results[I], ExpectedVal, I, "Test 6")) {
        ++Failures;
        break;
      }
    }

    free<SrcAllocKind>(USMMemSrc, Q);
    free<DstAllocKind>(USMMemDst1, Q);
    free<DstAllocKind>(USMMemDst2, Q);
  }

  // Test 7 - 2D copy from 2x2 grid of rectangles.
  {
    constexpr size_t SRC_ELEMS = 4 * RECT_WIDTH * RECT_HEIGHT;
    constexpr size_t DST_ELEMS = SRC_ELEMS / 4;

    T *USMMemSrc = allocate<T, SrcAllocKind>(SRC_ELEMS, Q);
    T *USMMemDst1 = allocate<T, DstAllocKind>(DST_ELEMS, Q);
    T *USMMemDst2 = allocate<T, DstAllocKind>(DST_ELEMS, Q);
    T *USMMemDst3 = allocate<T, DstAllocKind>(DST_ELEMS, Q);
    T *USMMemDst4 = allocate<T, DstAllocKind>(DST_ELEMS, Q);
    event SrcFillEvent =
        fill_with<SrcAllocKind>(Q, USMMemSrc, SRC_ELEMS, [=](size_t I) {
          return ((I / RECT_WIDTH) + (I / (2 * DST_ELEMS))) % 2 ? ExpectedVal2
                                                                : ExpectedVal1;
        });
    event Dst1MemsetEvent =
        memset<DstAllocKind>(Q, USMMemDst1, 0, DST_ELEMS * sizeof(T));
    event Dst2MemsetEvent =
        memset<DstAllocKind>(Q, USMMemDst2, 0, DST_ELEMS * sizeof(T));
    event Dst3MemsetEvent =
        memset<DstAllocKind>(Q, USMMemDst3, 0, DST_ELEMS * sizeof(T));
    event Dst4MemsetEvent =
        memset<DstAllocKind>(Q, USMMemDst4, 0, DST_ELEMS * sizeof(T));
    // Top left rectangle.
    event FirstMemcpyEvent = doMemcpy2D<T, PathKind>(
        Q, USMMemDst1, RECT_WIDTH * sizeof(T), USMMemSrc,
        2 * RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT,
        {SrcFillEvent, Dst1MemsetEvent, Dst2MemsetEvent, Dst3MemsetEvent,
         Dst4MemsetEvent});
    // Bottom right rectangle.
    event SecondMemcpyEvent = doMemcpy2D<T, PathKind>(
        Q, USMMemDst2, RECT_WIDTH * sizeof(T),
        USMMemSrc + 2 * DST_ELEMS + RECT_WIDTH, 2 * RECT_WIDTH * sizeof(T),
        RECT_WIDTH * sizeof(T), RECT_HEIGHT,
        {FirstMemcpyEvent, SrcFillEvent, Dst1MemsetEvent, Dst2MemsetEvent,
         Dst3MemsetEvent, Dst4MemsetEvent});
    // Bottom left rectangle.
    event ThirdMemcpyEvent = doMemcpy2D<T, PathKind>(
        Q, USMMemDst3, RECT_WIDTH * sizeof(T), USMMemSrc + 2 * DST_ELEMS,
        2 * RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT,
        {FirstMemcpyEvent, SecondMemcpyEvent, SrcFillEvent, Dst1MemsetEvent,
         Dst2MemsetEvent, Dst3MemsetEvent, Dst4MemsetEvent});
    // Top right rectangle.
    doMemcpy2D<T, PathKind>(
        Q, USMMemDst4, RECT_WIDTH * sizeof(T), USMMemSrc + RECT_WIDTH,
        2 * RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT,
        {FirstMemcpyEvent, SecondMemcpyEvent, ThirdMemcpyEvent, SrcFillEvent,
         Dst1MemsetEvent, Dst2MemsetEvent, Dst3MemsetEvent, Dst4MemsetEvent})
        .wait();
    std::vector<T> Results;
    Results.resize(SRC_ELEMS);
    copy_to_host<DstAllocKind>(Q, USMMemDst1, Results.data(), DST_ELEMS);
    copy_to_host<DstAllocKind>(Q, USMMemDst2, Results.data() + DST_ELEMS,
                               DST_ELEMS);
    copy_to_host<DstAllocKind>(Q, USMMemDst3, Results.data() + 2 * DST_ELEMS,
                               DST_ELEMS);
    copy_to_host<DstAllocKind>(Q, USMMemDst4, Results.data() + 3 * DST_ELEMS,
                               DST_ELEMS);
    Q.wait();

    for (size_t I = 0; I < SRC_ELEMS; ++I) {
      T ExpectedVal = I >= 2 * DST_ELEMS ? ExpectedVal2 : ExpectedVal1;
      if (!checkResult<SrcAllocKind, DstAllocKind, PathKind>(
              Results[I], ExpectedVal, I, "Test 7")) {
        ++Failures;
        break;
      }
    }

    free<SrcAllocKind>(USMMemSrc, Q);
    free<DstAllocKind>(USMMemDst1, Q);
    free<DstAllocKind>(USMMemDst2, Q);
    free<DstAllocKind>(USMMemDst3, Q);
    free<DstAllocKind>(USMMemDst4, Q);
  }

  return Failures;
}

template <typename T, Alloc SrcAllocKind, Alloc DstAllocKind>
int testForAllPaths(queue &Q, T ExpectedVal1, T ExpectedVal2) {
  int Failures = 0;
  Failures += test<T, SrcAllocKind, DstAllocKind, OperationPath::Expanded>(
      Q, ExpectedVal1, ExpectedVal2);
  Failures +=
      test<T, SrcAllocKind, DstAllocKind, OperationPath::ExpandedDependsOn>(
          Q, ExpectedVal1, ExpectedVal2);
  Failures +=
      test<T, SrcAllocKind, DstAllocKind, OperationPath::ShortcutNoEvent>(
          Q, ExpectedVal1, ExpectedVal2);
  Failures +=
      test<T, SrcAllocKind, DstAllocKind, OperationPath::ShortcutOneEvent>(
          Q, ExpectedVal1, ExpectedVal2);
  Failures +=
      test<T, SrcAllocKind, DstAllocKind, OperationPath::ShortcutEventList>(
          Q, ExpectedVal1, ExpectedVal2);
  return Failures;
}

template <Alloc SrcAllocKind, Alloc DstAllocKind>
int testForAllTypesAndPaths(queue &Q) {

  bool SupportsHalf = Q.get_device().has(aspect::fp16);
  bool SupportsDouble = Q.get_device().has(aspect::fp64);

  TestStruct TestStructRef1{42, 'f'}, TestStructRef2{1234, 'd'};

  int Failures = 0;
  Failures += testForAllPaths<char, SrcAllocKind, DstAllocKind>(Q, 'f', 'd');
  Failures += testForAllPaths<short, SrcAllocKind, DstAllocKind>(Q, 1234, 42);
  Failures += testForAllPaths<int, SrcAllocKind, DstAllocKind>(Q, 42, 1234);
  Failures += testForAllPaths<long, SrcAllocKind, DstAllocKind>(Q, 1242, 34);
  Failures +=
      testForAllPaths<long long, SrcAllocKind, DstAllocKind>(Q, 34, 1242);
  if (SupportsHalf)
    Failures += testForAllPaths<sycl::half, SrcAllocKind, DstAllocKind>(
        Q, 12.34f, 42.24f);
  Failures +=
      testForAllPaths<float, SrcAllocKind, DstAllocKind>(Q, 42.24f, 12.34f);
  if (SupportsDouble)
    Failures +=
        testForAllPaths<double, SrcAllocKind, DstAllocKind>(Q, 42.34, 12.24);
  Failures += testForAllPaths<TestStruct, SrcAllocKind, DstAllocKind>(
      Q, TestStructRef1, TestStructRef2);
  return Failures;
}

template <Alloc SrcAllocKind> int testForAllTypesAndPathsAndDsts(queue &Q) {
  int Failures = 0;
  Failures += testForAllTypesAndPaths<SrcAllocKind, Alloc::DirectHost>(Q);
  if (Q.get_device().has(aspect::usm_device_allocations))
    Failures += testForAllTypesAndPaths<SrcAllocKind, Alloc::Device>(Q);
  if (Q.get_device().has(aspect::usm_host_allocations))
    Failures += testForAllTypesAndPaths<SrcAllocKind, Alloc::Host>(Q);
  if (Q.get_device().has(aspect::usm_shared_allocations))
    Failures += testForAllTypesAndPaths<SrcAllocKind, Alloc::Shared>(Q);
  return Failures;
}

int main() {
  queue Q;

  int Failures = 0;
  Failures += testForAllTypesAndPathsAndDsts<Alloc::DirectHost>(Q);
  if (Q.get_device().has(aspect::usm_device_allocations))
    Failures += testForAllTypesAndPathsAndDsts<Alloc::Device>(Q);
  if (Q.get_device().has(aspect::usm_host_allocations))
    Failures += testForAllTypesAndPathsAndDsts<Alloc::Host>(Q);
  if (Q.get_device().has(aspect::usm_shared_allocations))
    Failures += testForAllTypesAndPathsAndDsts<Alloc::Shared>(Q);

  if (!Failures)
    std::cout << "Passed!" << std::endl;

  return Failures;
}
