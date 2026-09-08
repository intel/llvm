//==--------- executable_graph.hpp --- SYCL graph extension ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"                                  // for graph_state
#include <sycl/detail/export.hpp>                      // for __SYCL_EXPORT
#include <sycl/detail/owner_less_base.hpp>             // for OwnerLessBase
#include <sycl/ext/oneapi/experimental/graph/node.hpp> // for node class
#include <sycl/property_list.hpp>                      // for property_list

#include <cstdint> // for uint64_t
#include <memory>  // for shared_ptr
#include <vector>  // for vector

namespace sycl {
inline namespace _V1 {
// Forward declarations
class context;

namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {
// Forward declarations
class graph_impl;
class exec_graph_impl;

// Templateless executable command-graph base class.
class __SYCL_EXPORT executable_command_graph
    : public sycl::detail::OwnerLessBase<executable_command_graph> {
  friend sycl::detail::ImplUtils;

public:
  /// An executable command-graph is not user constructable.
  executable_command_graph() = delete;

  /// Update the inputs & output of the graph.
  /// @param Graph Graph to use the inputs and outputs of.
  void update(const command_graph<graph_state::modifiable> &Graph);

  /// Updates a single node in this graph based on the contents of the provided
  /// node.
  /// @param Node The node to use for updating the graph.
  void update(const node &Node);

  /// Updates a number of nodes in this graph based on the contents of the
  /// provided nodes.
  /// @param Nodes The nodes to use for updating the graph.
  void update(const std::vector<node> &Nodes);

  /// Return the total amount of memory required by this graph for graph-owned
  /// memory allocations.
  size_t get_required_mem_size() const;

  /// Queries a characteristic of this graph.
  ///
  /// Descriptors report the graph's effective state rather than the contents
  /// of the property list it was finalized with.
  /// @tparam Param One of the descriptors in the `info::graph` namespace which
  /// is valid in the executable state.
  /// @return The value of the queried characteristic.
  template <typename Param>
  typename Param::return_type get_info() const noexcept {
    static_assert(detail::is_graph_info_desc<Param>::value,
                  "Param must be one of the info descriptors in the "
                  "sycl::ext::oneapi::experimental::info::graph namespace.");
    static_assert(detail::graph_info_valid_in_executable<Param>::value,
                  "This info descriptor is not valid for a command_graph in "
                  "the graph_state::executable state.");
    return static_cast<typename Param::return_type>(
        getInfoImpl(Param::info_kind));
  }

  /// Common Reference Semantics
  friend bool operator==(const executable_command_graph &LHS,
                         const executable_command_graph &RHS) {
    return LHS.impl == RHS.impl;
  }
  friend bool operator!=(const executable_command_graph &LHS,
                         const executable_command_graph &RHS) {
    return !operator==(LHS, RHS);
  }

protected:
  /// Constructor used by internal runtime.
  /// @param Graph Detail implementation class to construct with.
  /// @param Ctx Context to use for graph.
  /// @param PropList Optional list of properties to pass.
  executable_command_graph(const std::shared_ptr<detail::graph_impl> &Graph,
                           const sycl::context &Ctx,
                           const property_list &PropList = {});

  /// Creates a backend representation of the graph in \p impl member variable.
  void finalizeImpl();

  /// Templateless implementation of get_info(). Every graph descriptor is
  /// scalar, so all of them share this one exported entry point and are
  /// returned widened to uint64_t.
  /// @param Kind Which characteristic to report.
  /// @return The characteristic's value, widened to uint64_t.
  uint64_t getInfoImpl(detail::graph_info_kind Kind) const noexcept;

  std::shared_ptr<detail::exec_graph_impl> impl;
};
} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
