//===-------------- nativecpu_state.hpp - SYCL Native CPU state -------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once
#include <cstdlib>
namespace native_cpu {

struct state {
  size_t MGlobal_id[3];
  size_t MGlobal_range[3];
  size_t MWorkGroup_size[3];
  size_t MWorkGroup_id[3];
  size_t MLocal_id[3];
  size_t MNumGroups[3];
  size_t MGlobalOffset[3];
  state(size_t globalR0, size_t globalR1, size_t globalR2, size_t localR0,
        size_t localR1, size_t localR2, size_t globalO0, size_t globalO1,
        size_t globalO2)
      : MGlobal_range{globalR0, globalR1, globalR2}, MWorkGroup_size{localR0,
                                                                     localR1,
                                                                     localR2},
        MNumGroups{globalR0 / localR0, globalR1 / localR1, globalR2 / localR2},
        MGlobalOffset{globalO0, globalO1, globalO2} {
    MGlobal_id[0] = 0;
    MGlobal_id[1] = 0;
    MGlobal_id[2] = 0;
    MWorkGroup_id[0] = 0;
    MWorkGroup_id[1] = 0;
    MWorkGroup_id[2] = 0;
    MLocal_id[0] = 0;
    MLocal_id[1] = 0;
    MLocal_id[2] = 0;
  }

  void update(size_t group0, size_t group1, size_t group2, size_t local0,
              size_t local1, size_t local2) {
    MWorkGroup_id[0] = group0;
    MWorkGroup_id[1] = group1;
    MWorkGroup_id[2] = group2;
    MLocal_id[0] = local0;
    MLocal_id[1] = local1;
    MLocal_id[2] = local2;
    MGlobal_id[0] =
        MWorkGroup_size[0] * MWorkGroup_id[0] + MLocal_id[0] + MGlobalOffset[0];
    MGlobal_id[1] =
        MWorkGroup_size[1] * MWorkGroup_id[1] + MLocal_id[1] + MGlobalOffset[1];
    MGlobal_id[2] =
        MWorkGroup_size[2] * MWorkGroup_id[2] + MLocal_id[2] + MGlobalOffset[2];
  }

  void update(size_t group0, size_t group1, size_t group2) {
    MWorkGroup_id[0] = group0;
    MWorkGroup_id[1] = group1;
    MWorkGroup_id[2] = group2;
  }
};

} // namespace native_cpu
