// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

// Checks the traits of the command_graph information descriptors

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <type_traits>

namespace exp_ext = sycl::ext::oneapi::experimental;
namespace exp_detail = sycl::ext::oneapi::experimental::detail;

// Descriptors are recognized as belonging to the graph info family.
static_assert(exp_detail::is_graph_info_desc<
              exp_ext::info::graph::recording_mode>::value);
static_assert(
    exp_detail::is_graph_info_desc<exp_ext::info::graph::updatable>::value);

// ... and descriptors of other families are not.
static_assert(!exp_detail::is_graph_info_desc<sycl::info::device::name>::value);

// Graph descriptors are not accepted by the other info families either.
static_assert(
    !sycl::detail::is_info_desc_for<exp_ext::info::graph::recording_mode,
                                    sycl::detail::info_class::device>::value);
static_assert(
    !sycl::detail::is_info_desc_for<exp_ext::info::graph::updatable,
                                    sycl::detail::info_class::device>::value);

// Per-state validity.
static_assert(
    exp_detail::graph_info_desc<exp_ext::info::graph::recording_mode,
                                exp_ext::graph_state::modifiable>::valid);
static_assert(
    exp_detail::graph_info_desc<exp_ext::info::graph::recording_mode,
                                exp_ext::graph_state::executable>::valid);
static_assert(
    !exp_detail::graph_info_desc<exp_ext::info::graph::updatable,
                                 exp_ext::graph_state::modifiable>::valid);
static_assert(
    exp_detail::graph_info_desc<exp_ext::info::graph::updatable,
                                exp_ext::graph_state::executable>::valid);

void positive(
    const exp_ext::command_graph<exp_ext::graph_state::modifiable> &G,
    const exp_ext::command_graph<exp_ext::graph_state::executable> &ExecG) {
  (void)G.get_info<exp_ext::info::graph::recording_mode>();
  (void)ExecG.get_info<exp_ext::info::graph::recording_mode>();
  (void)ExecG.get_info<exp_ext::info::graph::updatable>();
}

void updatable_on_modifiable(
    const exp_ext::command_graph<exp_ext::graph_state::modifiable> &G) {
  // expected-error@*:* {{This info descriptor is not valid for a graph in the modifiable state.}}
  (void)G.get_info<exp_ext::info::graph::updatable>();
}

void wrong_info_family(
    const exp_ext::command_graph<exp_ext::graph_state::executable> &ExecG) {
  // expected-error@*:* {{Param must be one of the info descriptors in the sycl::ext::oneapi::experimental::info::graph namespace.}}
  // expected-error@*:* {{This info descriptor is not valid for a graph in the executable state.}}
  (void)ExecG.get_info<sycl::info::device::max_compute_units>();
}
