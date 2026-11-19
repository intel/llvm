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

/// Which layer owns the graph's node model.
enum class graph_recording_mode {
  runtime, ///< The SYCL runtime owns the node model.
  native,  ///< Graph capture is delegated to the backend.
};

namespace info::graph {

/// Reports whether native recording is in effect for this graph. Reports
/// effective state, not property-list contents: a graph that records natively
/// only because `SYCL_GRAPH_FORCE_NATIVE_RECORDING=1` is set still reports
/// `graph_recording_mode::native`.
struct recording_mode {
  using return_type = graph_recording_mode;
};

/// Reports whether this executable graph can be updated, i.e. whether it was
/// finalized with the `property::graph::updatable` property.
struct updatable {
  using return_type = bool;
};

} // namespace info::graph

namespace detail {

/// Selects which characteristic the templateless `getInfoImpl()` entry points
/// of the two graph base classes should report. Every descriptor above is
/// scalar and is returned widened to `uint64_t`, so adding a descriptor costs
/// an enumerator rather than an exported symbol.
///
/// These values are ABI: never reorder or reuse an enumerator.
enum class graph_info_kind : uint32_t {
  recording_mode = 0,
  updatable = 1,
};

/// Maps an `info::graph` descriptor to its query kind, per graph state.
/// Specialized only for the (descriptor, state) pairs the spec declares valid,
/// so this is the single place recording which query applies in which state.
///
/// The primary template stays defined rather than left incomplete so that
/// `get_info()` can read `valid` and `kind` for an arbitrary `Param` and
/// diagnose it with a `static_assert` instead of a hard error.
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

/// Detects that `Param` is one of the `info::graph` descriptors, i.e. that it
/// is valid in at least one graph state. Lets `get_info()` tell a descriptor
/// from another family apart from one used in the wrong state.
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
