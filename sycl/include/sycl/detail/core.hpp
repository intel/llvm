//==------------ sycl.hpp - SYCL2020 standard header file ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// This is an ongoing experimental activity in its early stage. No code outside
// this project must rely on the behavior of this header file - keep using
// <sycl/sycl.hpp>.
//
// Short-term plan/action items (in no particular order):
//  * Update more tests to use this instead of full <sycl/sycl.hpp>.
//  * Refactor includes so that transitive dependencies don't bring as much as
//    they currently do.
//  * Determine what else should be included here.

#include <sycl/accessor.hpp>
#include <sycl/buffer.hpp>
#include <sycl/queue.hpp>
