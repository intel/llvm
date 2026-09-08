//==----------- graph_common.hpp --- SYCL graph properties -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/info_desc_traits.hpp> // for rt_traits_base, info_class

#include <cstdint>     // for uint32_t
#include <type_traits> // for false_type, void_t

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

namespace detail {

/// Selects which characteristic the templateless `getInfoImpl()` entry points
/// of the two graph base classes should report. Every descriptor defined below
/// is scalar and is returned widened to `uint64_t`, so adding a descriptor
/// costs an enumerator rather than an exported symbol.
///
/// These values are ABI: never reorder or reuse an enumerator.
enum class graph_info_kind : uint32_t {
  recording_mode = 0,
  updatable = 1,
};

} // namespace detail

namespace info::graph {

/// Reports whether native recording is in effect for this graph. Reports
/// effective state, not property-list contents: a graph that records natively
/// only because `SYCL_GRAPH_FORCE_NATIVE_RECORDING=1` is set still reports
/// `graph_recording_mode::native`.
struct recording_mode
    : sycl::detail::rt_traits_base<sycl::detail::info_class::graph> {
  using return_type = graph_recording_mode;
  static constexpr auto info_kind =
      experimental::detail::graph_info_kind::recording_mode;
  static constexpr bool valid_in_modifiable = true;
  static constexpr bool valid_in_executable = true;
};

/// Reports whether this executable graph can be updated, i.e. whether it was
/// finalized with the `property::graph::updatable` property.
struct updatable
    : sycl::detail::rt_traits_base<sycl::detail::info_class::graph> {
  using return_type = bool;
  static constexpr auto info_kind =
      experimental::detail::graph_info_kind::updatable;
  static constexpr bool valid_in_modifiable = false;
  static constexpr bool valid_in_executable = true;
};

} // namespace info::graph

namespace detail {

/// Detects that `Param` is one of the `info::graph` descriptors above. Also
/// checks for the per-state validity members so a descriptor missing them
/// fails the family check rather than erroring inside `get_info()`.
template <typename Param, typename = void>
struct is_graph_info_desc : std::false_type {};

template <typename Param>
struct is_graph_info_desc<Param,
                          std::void_t<decltype(Param::info_kind),
                                      decltype(Param::valid_in_modifiable),
                                      decltype(Param::valid_in_executable)>>
    : sycl::detail::is_info_desc_for<Param, sycl::detail::info_class::graph> {};

// Read a descriptor's per-state validity without hard-erroring on a type that
// is not a graph descriptor at all, so that `get_info()` diagnoses the family
// mismatch instead of a missing member.
template <typename Param, typename = void>
struct graph_info_valid_in_modifiable : std::false_type {};

template <typename Param>
struct graph_info_valid_in_modifiable<
    Param, std::enable_if_t<Param::valid_in_modifiable>> : std::true_type {};

template <typename Param, typename = void>
struct graph_info_valid_in_executable : std::false_type {};

template <typename Param>
struct graph_info_valid_in_executable<
    Param, std::enable_if_t<Param::valid_in_executable>> : std::true_type {};

} // namespace detail

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
