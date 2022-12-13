//==---- memset2d.cpp - USM 2D memset test ---------------------------------==//
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

template <OperationPath PathKind>
event doMemset2D(queue &Q, void *Dest, size_t DestPitch, int Value,
                 size_t Width, size_t Height, std::vector<event> DepEvents) {
  if constexpr (PathKind == OperationPath::Expanded) {
    sycl::event::wait(DepEvents);
    return Q.submit([&](handler &CGH) {
      CGH.ext_oneapi_memset2d(Dest, DestPitch, Value, Width, Height);
    });
  }
  if constexpr (PathKind == OperationPath::ExpandedDependsOn) {
    return Q.submit([&](handler &CGH) {
      CGH.depends_on(DepEvents);
      CGH.ext_oneapi_memset2d(Dest, DestPitch, Value, Width, Height);
    });
  }
  if constexpr (PathKind == OperationPath::ShortcutNoEvent) {
    sycl::event::wait(DepEvents);
    return Q.ext_oneapi_memset2d(Dest, DestPitch, Value, Width, Height);
  }
  if constexpr (PathKind == OperationPath::ShortcutOneEvent) {
    assert(DepEvents.size() && "No events in dependencies!");
    // wait on all other events than the first.
    for (size_t I = 1; I < DepEvents.size(); ++I)
      DepEvents[I].wait();
    return Q.ext_oneapi_memset2d(Dest, DestPitch, Value, Width, Height,
                                 DepEvents[0]);
  }
  if constexpr (PathKind == OperationPath::ShortcutEventList) {
    return Q.ext_oneapi_memset2d(Dest, DestPitch, Value, Width, Height,
                                 DepEvents);
  }
}

template <usm::alloc AllocKind, OperationPath PathKind>
int test(queue &Q, unsigned char ExpectedVal1, unsigned char ExpectedVal2) {
  int Failures = 0;

  // Test 1 - 2D memset entire buffer.
  {
    constexpr size_t DST_ELEMS = RECT_WIDTH * RECT_HEIGHT;

    unsigned char *USMMemDst =
        sycl::malloc<unsigned char>(DST_ELEMS, Q, AllocKind);
    event DstMemsetEvent = Q.memset(USMMemDst, 0, DST_ELEMS);
    doMemset2D<PathKind>(Q, USMMemDst, RECT_WIDTH, ExpectedVal1, RECT_WIDTH,
                         RECT_HEIGHT, {DstMemsetEvent})
        .wait();
    std::vector<unsigned char> Results;
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

  // Test 2 - 2D memset vertically adjacent regions.
  {
    constexpr size_t DST_ELEMS = 2 * RECT_WIDTH * RECT_HEIGHT;

    unsigned char *USMMemDst =
        sycl::malloc<unsigned char>(DST_ELEMS, Q, AllocKind);
    event DstMemsetEvent = Q.memset(USMMemDst, 0, DST_ELEMS);
    event FirstFillEvent =
        doMemset2D<PathKind>(Q, USMMemDst, RECT_WIDTH, ExpectedVal1, RECT_WIDTH,
                             RECT_HEIGHT, {DstMemsetEvent});
    doMemset2D<PathKind>(Q, USMMemDst + DST_ELEMS / 2, RECT_WIDTH, ExpectedVal2,
                         RECT_WIDTH, RECT_HEIGHT,
                         {FirstFillEvent, DstMemsetEvent})
        .wait();
    std::vector<unsigned char> Results;
    Results.resize(DST_ELEMS);
    Q.copy(USMMemDst, Results.data(), DST_ELEMS).wait();

    for (size_t I = 0; I < DST_ELEMS; ++I) {
      unsigned char ExpectedVal =
          I >= (DST_ELEMS / 2) ? ExpectedVal2 : ExpectedVal1;
      if (!checkResult<AllocKind, PathKind>(Results[I], ExpectedVal, I,
                                            "Test 2")) {
        ++Failures;
        break;
      }
    }

    sycl::free(USMMemDst, Q);
  }

  // Test 3 - 2D memset horizontally adjacent regions.
  {
    constexpr size_t DST_ELEMS = 2 * RECT_WIDTH * RECT_HEIGHT;

    unsigned char *USMMemDst =
        sycl::malloc<unsigned char>(DST_ELEMS, Q, AllocKind);
    event DstMemsetEvent = Q.memset(USMMemDst, 0, DST_ELEMS);
    event FirstFillEvent =
        doMemset2D<PathKind>(Q, USMMemDst, 2 * RECT_WIDTH, ExpectedVal1,
                             RECT_WIDTH, RECT_HEIGHT, {DstMemsetEvent});
    doMemset2D<PathKind>(Q, USMMemDst + RECT_WIDTH, 2 * RECT_WIDTH,
                         ExpectedVal2, RECT_WIDTH, RECT_HEIGHT,
                         {FirstFillEvent, DstMemsetEvent})
        .wait();
    std::vector<unsigned char> Results;
    Results.resize(DST_ELEMS);
    Q.copy(USMMemDst, Results.data(), DST_ELEMS).wait();

    for (size_t I = 0; I < DST_ELEMS; ++I) {
      unsigned char ExpectedVal =
          (I / RECT_WIDTH) % 2 ? ExpectedVal2 : ExpectedVal1;
      if (!checkResult<AllocKind, PathKind>(Results[I], ExpectedVal, I,
                                            "Test 3")) {
        ++Failures;
        break;
      }
    }

    sycl::free(USMMemDst, Q);
  }

  // Test 4 - 2D memset 2x2 grid of rectangles.
  {
    constexpr size_t DST_ELEMS = 4 * RECT_WIDTH * RECT_HEIGHT;

    unsigned char *USMMemDst =
        sycl::malloc<unsigned char>(DST_ELEMS, Q, AllocKind);
    event DstMemsetEvent = Q.memset(USMMemDst, 0, DST_ELEMS);
    // Top left rectangle.
    event FirstFillEvent =
        doMemset2D<PathKind>(Q, USMMemDst, 2 * RECT_WIDTH, ExpectedVal1,
                             RECT_WIDTH, RECT_HEIGHT, {DstMemsetEvent});
    // Top right rectangle.
    event SecondFillEvent = doMemset2D<PathKind>(
        Q, USMMemDst + RECT_WIDTH, 2 * RECT_WIDTH, ExpectedVal2, RECT_WIDTH,
        RECT_HEIGHT, {FirstFillEvent, DstMemsetEvent});
    // Bottom left rectangle.
    event ThirdFillEvent = doMemset2D<PathKind>(
        Q, USMMemDst + DST_ELEMS / 2, 2 * RECT_WIDTH, ExpectedVal2, RECT_WIDTH,
        RECT_HEIGHT, {FirstFillEvent, SecondFillEvent, DstMemsetEvent});
    // Bottom right rectangle.
    doMemset2D<PathKind>(
        Q, USMMemDst + DST_ELEMS / 2 + RECT_WIDTH, 2 * RECT_WIDTH, ExpectedVal1,
        RECT_WIDTH, RECT_HEIGHT,
        {FirstFillEvent, SecondFillEvent, ThirdFillEvent, DstMemsetEvent})
        .wait();
    std::vector<unsigned char> Results;
    Results.resize(DST_ELEMS);
    Q.copy(USMMemDst, Results.data(), DST_ELEMS).wait();

    for (size_t I = 0; I < DST_ELEMS; ++I) {
      unsigned char ExpectedVal = ((I / RECT_WIDTH) + (I / (DST_ELEMS / 2))) % 2
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

template <usm::alloc AllocKind>
int testForAllPaths(queue &Q, unsigned char ExpectedVal1,
                    unsigned char ExpectedVal2) {
  int Failures = 0;
  Failures +=
      test<AllocKind, OperationPath::Expanded>(Q, ExpectedVal1, ExpectedVal2);
  Failures += test<AllocKind, OperationPath::ExpandedDependsOn>(Q, ExpectedVal1,
                                                                ExpectedVal2);
  Failures += test<AllocKind, OperationPath::ShortcutNoEvent>(Q, ExpectedVal1,
                                                              ExpectedVal2);
  Failures += test<AllocKind, OperationPath::ShortcutOneEvent>(Q, ExpectedVal1,
                                                               ExpectedVal2);
  Failures += test<AllocKind, OperationPath::ShortcutEventList>(Q, ExpectedVal1,
                                                                ExpectedVal2);
  return Failures;
}

int main() {
  queue Q;

  int Failures = 0;
  if (Q.get_device().has(aspect::usm_device_allocations))
    Failures += testForAllPaths<usm::alloc::device>(Q, 123, 42);
  if (Q.get_device().has(aspect::usm_host_allocations))
    Failures += testForAllPaths<usm::alloc::host>(Q, 123, 42);
  if (Q.get_device().has(aspect::usm_shared_allocations))
    Failures += testForAllPaths<usm::alloc::shared>(Q, 123, 42);

  if (!Failures)
    std::cout << "Passed!" << std::endl;

  return Failures;
}
