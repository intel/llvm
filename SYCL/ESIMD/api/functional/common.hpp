//===-- common.hpp - This file provides common functions for simd constructors
//      tests -------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Common file for test on simd class.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include "../../esimd_test_utils.hpp"
#include "logger.hpp"
#include "type_coverage.hpp"
