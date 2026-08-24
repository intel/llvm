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

#include <memory> // for shared_ptr
#include <vector> // for vector

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

  template <class Obj>
  friend const decltype(Obj::impl) &
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);

  /// Creates a backend representation of the graph in \p impl member variable.
  void finalizeImpl();

  std::shared_ptr<detail::exec_graph_impl> impl;
};
} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
