// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

// Checks the traits of the command_graph information descriptors and the
// per-state validity diagnostics of command_graph::get_info().

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <type_traits>

namespace exp_ext = sycl::ext::oneapi::experimental;
namespace info_class = sycl::detail::info_class;

// Descriptors belong to the graph info class.
static_assert(sycl::detail::is_info_desc_for<
              exp_ext::info::graph::recording_mode, info_class::graph>::value);
static_assert(sycl::detail::is_info_desc_for<exp_ext::info::graph::updatable,
                                             info_class::graph>::value);

// ... and not to any other.
static_assert(!sycl::detail::is_info_desc_for<
              exp_ext::info::graph::recording_mode, info_class::device>::value);

// Return types are as declared.
static_assert(std::is_same_v<exp_ext::info::graph::recording_mode::return_type,
                             exp_ext::graph_recording_mode>);
static_assert(
    std::is_same_v<exp_ext::info::graph::updatable::return_type, bool>);

// Per-state validity.
static_assert(exp_ext::info::graph::recording_mode::valid_in_modifiable);
static_assert(exp_ext::info::graph::recording_mode::valid_in_executable);
static_assert(!exp_ext::info::graph::updatable::valid_in_modifiable);
static_assert(exp_ext::info::graph::updatable::valid_in_executable);

void positive(
    const exp_ext::command_graph<exp_ext::graph_state::modifiable> &G,
    const exp_ext::command_graph<exp_ext::graph_state::executable> &ExecG) {
  (void)G.get_info<exp_ext::info::graph::recording_mode>();
  (void)ExecG.get_info<exp_ext::info::graph::recording_mode>();
  (void)ExecG.get_info<exp_ext::info::graph::updatable>();
}

void updatable_on_modifiable(
    const exp_ext::command_graph<exp_ext::graph_state::modifiable> &G) {
  // expected-error@*:* {{This info descriptor is not valid for a command_graph in the graph_state::modifiable state.}}
  (void)G.get_info<exp_ext::info::graph::updatable>();
}
