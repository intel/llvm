// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

// Checks the traits of the command_graph information descriptors and the
// per-state validity diagnostics of command_graph::get_info().

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

// Return types are as declared.
static_assert(std::is_same_v<exp_ext::info::graph::recording_mode::return_type,
                             exp_ext::graph_recording_mode>);
static_assert(
    std::is_same_v<exp_ext::info::graph::updatable::return_type, bool>);

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
  // expected-error@*:* {{This info descriptor is not valid for a command_graph in the graph_state::modifiable state.}}
  (void)G.get_info<exp_ext::info::graph::updatable>();
}

void wrong_info_family(
    const exp_ext::command_graph<exp_ext::graph_state::executable> &ExecG) {
  // expected-error@*:* {{Param must be one of the info descriptors in the sycl::ext::oneapi::experimental::info::graph namespace.}}
  // expected-error@*:* {{This info descriptor is not valid for a command_graph in the graph_state::executable state.}}
  // A descriptor with a scalar return_type, so that the two static_asserts
  // above are the only diagnostics and the test does not depend on how
  // getInfoImpl()'s uint64_t result is converted.
  (void)ExecG.get_info<sycl::info::device::max_compute_units>();
}
