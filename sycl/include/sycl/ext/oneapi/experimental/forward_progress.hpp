//==------- forward_progress.hpp - sycl_ext_oneapi_forward_progress -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

enum class forward_progress_guarantee { concurrent, parallel, weakly_parallel };

enum class execution_scope {
  work_item,
  sub_group,
  work_group,
  root_group,
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
