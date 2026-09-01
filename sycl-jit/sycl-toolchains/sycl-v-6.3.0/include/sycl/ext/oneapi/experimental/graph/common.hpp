//==----------- graph_common.hpp --- SYCL graph properties -----------------==//
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

/// State to template the command_graph class on.
enum class graph_state {
  modifiable, ///< In modifiable state, commands can be added to graph.
  executable, ///< In executable state, the graph is ready to execute.
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
