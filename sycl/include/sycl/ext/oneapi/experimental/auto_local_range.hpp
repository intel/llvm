//==--- auto_local_range.hpp --- SYCL extension for auto range -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/range.hpp>

#define SYCL_EXT_ONEAPI_AUTO_LOCAL_RANGE 1

namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {

template <int Dimensions> static const inline range<Dimensions> auto_range;

template <> static const inline range<3> auto_range<3>{0, 0, 0};

template <> static const inline range<2> auto_range<2>{0, 0};

template <> static const inline range<1> auto_range<1>{0};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
