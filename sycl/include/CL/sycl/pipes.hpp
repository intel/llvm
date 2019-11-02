//==---------------- pipes.hpp - SYCL pipes ------------*- C++ -*-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/intel/pipes.hpp>

namespace cl {
namespace sycl {
template <class name, class dataT, int32_t min_capacity = 0>
using pipe = intel::pipe<name, dataT, min_capacity>;
} // namespace sycl
} // namespace cl
