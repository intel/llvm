//===--------- ur_level_zero_kernel.hpp - Level Zero Adapter ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include "ur_level_zero_common.hpp"

struct _ur_kernel_handle_t : _pi_object {
  _ur_kernel_handle_t() {}
};
