//==---- memset2d.cpp - USM 2D memset test ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include "memops2d_utils.hpp"

using namespace sycl;

constexpr size_t RECT_WIDTH = 50;
constexpr size_t RECT_HEIGHT = 21;

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
  if constexpr (PathKind == OperationPath::ShortcutEventList) {
    return Q.ext_oneapi_memset2d(Dest, DestPitch, Value, Width, Height,
                                 DepEvents);
  }
}

template <Alloc AllocKind, OperationPath PathKind>
int test(queue &Q, unsigned char ExpectedVal1, unsigned char ExpectedVal2) {
  int Failures = 0;

  // Test 1 - 2D memset entire buffer.
  {
    constexpr size_t DST_ELEMS = RECT_WIDTH * RECT_HEIGHT;

    unsigned char *USMMemDst = allocate<unsigned char, AllocKind>(DST_ELEMS, Q);
    event DstMemsetEvent = memset<AllocKind>(Q, USMMemDst, 0, DST_ELEMS);
    doMemset2D<PathKind>(Q, USMMemDst, RECT_WIDTH, ExpectedVal1, RECT_WIDTH,
                         RECT_HEIGHT, {DstMemsetEvent})
        .wait();
    std::vector<unsigned char> Results;
    Results.resize(DST_ELEMS);
    copy_to_host<AllocKind>(Q, USMMemDst, Results.data(), DST_ELEMS).wait();

    for (size_t I = 0; I < DST_ELEMS; ++I) {
      if (!checkResult<AllocKind, PathKind>(Results[I], ExpectedVal1, I,
                                            "Test 1")) {
        ++Failures;
        break;
      }
    }

    free<AllocKind>(USMMemDst, Q);
  }

  // Test 2 - 2D memset vertically adjacent regions.
  {
    constexpr size_t DST_ELEMS = 2 * RECT_WIDTH * RECT_HEIGHT;

    unsigned char *USMMemDst = allocate<unsigned char, AllocKind>(DST_ELEMS, Q);
    event DstMemsetEvent = memset<AllocKind>(Q, USMMemDst, 0, DST_ELEMS);
    event FirstFillEvent =
        doMemset2D<PathKind>(Q, USMMemDst, RECT_WIDTH, ExpectedVal1, RECT_WIDTH,
                             RECT_HEIGHT, {DstMemsetEvent});
    doMemset2D<PathKind>(Q, USMMemDst + DST_ELEMS / 2, RECT_WIDTH, ExpectedVal2,
                         RECT_WIDTH, RECT_HEIGHT,
                         {FirstFillEvent, DstMemsetEvent})
        .wait();
    std::vector<unsigned char> Results;
    Results.resize(DST_ELEMS);
    copy_to_host<AllocKind>(Q, USMMemDst, Results.data(), DST_ELEMS).wait();

    for (size_t I = 0; I < DST_ELEMS; ++I) {
      unsigned char ExpectedVal =
          I >= (DST_ELEMS / 2) ? ExpectedVal2 : ExpectedVal1;
      if (!checkResult<AllocKind, PathKind>(Results[I], ExpectedVal, I,
                                            "Test 2")) {
        ++Failures;
        break;
      }
    }

    free<AllocKind>(USMMemDst, Q);
  }

  // Test 3 - 2D memset horizontally adjacent regions.
  {
    constexpr size_t DST_ELEMS = 2 * RECT_WIDTH * RECT_HEIGHT;

    unsigned char *USMMemDst = allocate<unsigned char, AllocKind>(DST_ELEMS, Q);
    event DstMemsetEvent = memset<AllocKind>(Q, USMMemDst, 0, DST_ELEMS);
    event FirstFillEvent =
        doMemset2D<PathKind>(Q, USMMemDst, 2 * RECT_WIDTH, ExpectedVal1,
                             RECT_WIDTH, RECT_HEIGHT, {DstMemsetEvent});
    doMemset2D<PathKind>(Q, USMMemDst + RECT_WIDTH, 2 * RECT_WIDTH,
                         ExpectedVal2, RECT_WIDTH, RECT_HEIGHT,
                         {FirstFillEvent, DstMemsetEvent})
        .wait();
    std::vector<unsigned char> Results;
    Results.resize(DST_ELEMS);
    copy_to_host<AllocKind>(Q, USMMemDst, Results.data(), DST_ELEMS).wait();

    for (size_t I = 0; I < DST_ELEMS; ++I) {
      unsigned char ExpectedVal =
          (I / RECT_WIDTH) % 2 ? ExpectedVal2 : ExpectedVal1;
      if (!checkResult<AllocKind, PathKind>(Results[I], ExpectedVal, I,
                                            "Test 3")) {
        ++Failures;
        break;
      }
    }

    free<AllocKind>(USMMemDst, Q);
  }

  // Test 4 - 2D memset 2x2 grid of rectangles.
  {
    constexpr size_t DST_ELEMS = 4 * RECT_WIDTH * RECT_HEIGHT;

    unsigned char *USMMemDst = allocate<unsigned char, AllocKind>(DST_ELEMS, Q);
    event DstMemsetEvent = memset<AllocKind>(Q, USMMemDst, 0, DST_ELEMS);
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
    copy_to_host<AllocKind>(Q, USMMemDst, Results.data(), DST_ELEMS).wait();

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

    free<AllocKind>(USMMemDst, Q);
  }

  return Failures;
}

template <Alloc AllocKind>
int testForAllPaths(queue &Q, unsigned char ExpectedVal1,
                    unsigned char ExpectedVal2) {
  int Failures = 0;
  Failures +=
      test<AllocKind, OperationPath::Expanded>(Q, ExpectedVal1, ExpectedVal2);
  Failures += test<AllocKind, OperationPath::ExpandedDependsOn>(Q, ExpectedVal1,
                                                                ExpectedVal2);
  Failures += test<AllocKind, OperationPath::ShortcutEventList>(Q, ExpectedVal1,
                                                                ExpectedVal2);
  return Failures;
}

int main() {
  queue Q;

  int Failures = 0;
  Failures += testForAllPaths<Alloc::DirectHost>(Q, 123, 42);
  if (Q.get_device().has(aspect::usm_device_allocations))
    Failures += testForAllPaths<Alloc::Device>(Q, 123, 42);
  if (Q.get_device().has(aspect::usm_host_allocations))
    Failures += testForAllPaths<Alloc::Host>(Q, 123, 42);
  if (Q.get_device().has(aspect::usm_shared_allocations))
    Failures += testForAllPaths<Alloc::Shared>(Q, 123, 42);

  if (!Failures)
    std::cout << "Passed!" << std::endl;

  return Failures;
}
