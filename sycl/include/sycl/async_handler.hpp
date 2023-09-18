//===- async_handler.hpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <functional> // for function

namespace sycl {
inline namespace _V1 {

// Forward declaration
class exception_list;

using async_handler = std::function<void(sycl::exception_list)>;
} // namespace _V1
} // namespace sycl
