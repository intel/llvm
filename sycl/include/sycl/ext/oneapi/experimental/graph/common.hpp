//==----------- graph_common.hpp --- SYCL graph properties -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>     // for uint32_t
#include <type_traits> // for bool_constant

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

/// Which layer owns manages the graph's nodes and edges
enum class graph_recording_mode {
  runtime, ///< The SYCL runtime owns the model.
  native,   ///< Graph capture is delegated to the backend.
};

namespace info::graph {

/// Reports whether native recording is in effect for this graph.
struct recording_mode {
  using return_type = graph_recording_mode;
};

/// Reports whether this executable graph can be updated
struct updatable {
  using return_type = bool;
};

} // namespace info::graph

namespace detail {

/// The two types of items that can be queried with command_graph::get_info()
enum class graph_info_kind : uint32_t {
  recording_mode = 0,
  updatable = 1,
};

/// Maps an info::graph descriptors to its query kind
template <typename Param, graph_state State> struct graph_info_desc {
  static constexpr bool valid = false;
  static constexpr graph_info_kind kind{};
};

template <graph_state State>
struct graph_info_desc<info::graph::recording_mode, State> {
  static constexpr bool valid = true;
  static constexpr graph_info_kind kind = graph_info_kind::recording_mode;
};

template <>
struct graph_info_desc<info::graph::updatable, graph_state::executable> {
  static constexpr bool valid = true;
  static constexpr graph_info_kind kind = graph_info_kind::updatable;
};

template <typename Param>
struct is_graph_info_desc
    : std::bool_constant<
          graph_info_desc<Param, graph_state::modifiable>::valid ||
          graph_info_desc<Param, graph_state::executable>::valid> {};

} // namespace detail

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
