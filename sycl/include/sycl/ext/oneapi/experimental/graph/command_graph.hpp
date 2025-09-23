//==--------- command_graph.hpp --- SYCL graph extension -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp" // for graph_state
#include "executable_graph.hpp"
#include "modifiable_graph.hpp"

#include <functional> // for function
#include <memory>     // for shared_ptr

namespace sycl {
inline namespace _V1 {
// Forward declarations
class queue;
class device;
class context;
class property_list;

namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {
// Forward declarations
class graph_impl;

// List of sycl features and extensions which are not supported by graphs. Used
// for throwing errors when these features are used with graphs.
enum class UnsupportedGraphFeatures {
  sycl_reductions = 0,
  sycl_specialization_constants = 1,
  sycl_kernel_bundle = 2,
  sycl_ext_oneapi_kernel_properties = 3,
  sycl_ext_oneapi_enqueue_barrier = 4,
  sycl_ext_oneapi_memcpy2d = 5,
  sycl_ext_oneapi_device_global = 6,
  sycl_ext_oneapi_bindless_images = 7,
  sycl_ext_oneapi_experimental_cuda_cluster_launch = 8,
  sycl_ext_codeplay_enqueue_native_command = 9,
  sycl_ext_oneapi_work_group_scratch_memory = 10,
  sycl_ext_oneapi_async_alloc = 11
};

inline const char *
UnsupportedFeatureToString(UnsupportedGraphFeatures Feature) {
  using UGF = UnsupportedGraphFeatures;
  switch (Feature) {
  case UGF::sycl_reductions:
    return "Reductions";
  case UGF::sycl_specialization_constants:
    return "Specialization Constants";
  case UGF::sycl_kernel_bundle:
    return "Kernel Bundles";
  case UGF::sycl_ext_oneapi_kernel_properties:
    return "sycl_ext_oneapi_kernel_properties";
  case UGF::sycl_ext_oneapi_enqueue_barrier:
    return "sycl_ext_oneapi_enqueue_barrier";
  case UGF::sycl_ext_oneapi_memcpy2d:
    return "sycl_ext_oneapi_memcpy2d";
  case UGF::sycl_ext_oneapi_device_global:
    return "sycl_ext_oneapi_device_global";
  case UGF::sycl_ext_oneapi_bindless_images:
    return "sycl_ext_oneapi_bindless_images";
  case UGF::sycl_ext_oneapi_experimental_cuda_cluster_launch:
    return "sycl_ext_oneapi_experimental_cuda_cluster_launch";
  case UGF::sycl_ext_codeplay_enqueue_native_command:
    return "sycl_ext_codeplay_enqueue_native_command";
  case UGF::sycl_ext_oneapi_work_group_scratch_memory:
    return "sycl_ext_oneapi_work_group_scratch_memory";
  case UGF::sycl_ext_oneapi_async_alloc:
    return "sycl_ext_oneapi_async_alloc";
  }

  assert(false && "Unhandled graphs feature");
  return {};
}
} // namespace detail

/// Graph in the modifiable state.
template <graph_state State = graph_state::modifiable>
class command_graph : public detail::modifiable_command_graph {
public:
  /// Constructor.
  /// @param SyclContext Context to use for graph.
  /// @param SyclDevice Device all nodes will be associated with.
  /// @param PropList Optional list of properties to pass.
  command_graph(const context &SyclContext, const device &SyclDevice,
                const property_list &PropList = {})
      : modifiable_command_graph(SyclContext, SyclDevice, PropList) {}

  /// Constructor.
  /// @param SyclQueue Queue to use for the graph device and context.
  /// @param PropList Optional list of properties to pass.
  explicit command_graph(const queue &SyclQueue,
                         const property_list &PropList = {})
      : modifiable_command_graph(SyclQueue, PropList) {}

  /// Constructor with default context.
  /// @param SyclDevice Device all nodes will be associated with.
  /// @param PropList Optional list of properties to pass.
  explicit command_graph(const device &SyclDevice,
                         const property_list &PropList = {})
      : modifiable_command_graph(SyclDevice, PropList) {}

private:
  /// Constructor used internally by the runtime.
  /// @param Impl Detail implementation class to construct object with.
  command_graph(const std::shared_ptr<detail::graph_impl> &Impl)
      : modifiable_command_graph(Impl) {}

  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(
      std::add_rvalue_reference_t<decltype(T::impl)> ImplObj);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(
      std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj);
};

template <>
class command_graph<graph_state::executable>
    : public detail::executable_command_graph {
protected:
  friend command_graph<graph_state::executable>
  detail::modifiable_command_graph::finalize(const sycl::property_list &) const;
  using detail::executable_command_graph::executable_command_graph;
};

/// Additional CTAD deduction guides.
template <graph_state State = graph_state::modifiable>
command_graph(const context &SyclContext, const device &SyclDevice,
              const property_list &PropList) -> command_graph<State>;

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
