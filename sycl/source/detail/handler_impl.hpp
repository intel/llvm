//==---------------- handler_impl.hpp - SYCL handler -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "sycl/handler.hpp"
#include <detail/cg.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <memory>
#include <sycl/ext/oneapi/experimental/graph.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental::detail {
class dynamic_parameter_impl;
}
namespace detail {

using KernelBundleImplPtr = std::shared_ptr<detail::kernel_bundle_impl>;

enum class HandlerSubmissionState : std::uint8_t {
  NO_STATE = 0,
  EXPLICIT_KERNEL_BUNDLE_STATE,
  SPEC_CONST_SET_STATE,
};

class handler_impl {
public:
  handler_impl(std::shared_ptr<queue_impl> SubmissionPrimaryQueue,
               std::shared_ptr<queue_impl> SubmissionSecondaryQueue,
               bool EventNeeded)
      : MSubmissionPrimaryQueue(std::move(SubmissionPrimaryQueue)),
        MSubmissionSecondaryQueue(std::move(SubmissionSecondaryQueue)),
        MEventNeeded(EventNeeded) {};

  handler_impl(
      std::shared_ptr<ext::oneapi::experimental::detail::graph_impl> Graph)
      : MGraph{Graph} {}

  handler_impl() = default;

  void setStateExplicitKernelBundle() {
    if (MSubmissionState == HandlerSubmissionState::SPEC_CONST_SET_STATE)
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Kernel bundle cannot be explicitly set after a specialization "
          "constant has been set");
    MSubmissionState = HandlerSubmissionState::EXPLICIT_KERNEL_BUNDLE_STATE;
  }

  void setStateSpecConstSet() {
    if (MSubmissionState ==
        HandlerSubmissionState::EXPLICIT_KERNEL_BUNDLE_STATE)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Specialization constants cannot be set after "
                            "explicitly setting the used kernel bundle");
    MSubmissionState = HandlerSubmissionState::SPEC_CONST_SET_STATE;
  }

  bool isStateExplicitKernelBundle() const {
    return MSubmissionState ==
           HandlerSubmissionState::EXPLICIT_KERNEL_BUNDLE_STATE;
  }

  /// Registers mutually exclusive submission states.
  HandlerSubmissionState MSubmissionState = HandlerSubmissionState::NO_STATE;

  /// Shared pointer to the primary queue implementation. This is different from
  /// the queue associated with the handler if the corresponding submission is
  /// a fallback from a previous submission.
  std::shared_ptr<queue_impl> MSubmissionPrimaryQueue;

  /// Shared pointer to the secondary queue implementation. Nullptr if no
  /// secondary queue fallback was given in the associated submission. This is
  /// equal to the queue associated with the handler if the corresponding
  /// submission is a fallback from a previous submission.
  std::shared_ptr<queue_impl> MSubmissionSecondaryQueue;

  /// Bool stores information about whether the event resulting from the
  /// corresponding work is required.
  bool MEventNeeded = true;

  // Stores auxiliary resources used by internal operations.
  std::vector<std::shared_ptr<const void>> MAuxiliaryResources;

  std::shared_ptr<detail::kernel_bundle_impl> MKernelBundle;

  ur_usm_advice_flags_t MAdvice = 0;

  // 2D memory operation information.
  size_t MSrcPitch = 0;
  size_t MDstPitch = 0;
  size_t MWidth = 0;
  size_t MHeight = 0;

  /// Offset into a device_global for copy operations.
  size_t MOffset = 0;
  /// Boolean flag for whether the device_global had the device_image_scope
  /// property.
  bool MIsDeviceImageScoped = false;

  // Program scope pipe information.

  // Pipe name that uniquely identifies a pipe.
  std::string HostPipeName;
  // Pipe host pointer, the address of its constexpr __pipe member.
  void *HostPipePtr = nullptr;
  // Host pipe read write operation is blocking.
  bool HostPipeBlocking = false;
  // The size of returned type for each read.
  size_t HostPipeTypeSize = 0;
  // If the pipe operation is read or write, 1 for read 0 for write.
  bool HostPipeRead = true;

  ur_kernel_cache_config_t MKernelCacheConfig = UR_KERNEL_CACHE_CONFIG_DEFAULT;

  bool MKernelIsCooperative = false;
  bool MKernelUsesClusterLaunch = false;

  // Extra information for bindless image copy
  ur_image_desc_t MSrcImageDesc = {};
  ur_image_desc_t MDstImageDesc = {};
  ur_image_format_t MSrcImageFormat = {};
  ur_image_format_t MDstImageFormat = {};
  ur_exp_image_copy_flags_t MImageCopyFlags = {};

  ur_rect_offset_t MSrcOffset = {};
  ur_rect_offset_t MDestOffset = {};
  ur_rect_region_t MCopyExtent = {};

  // Extra information for semaphore interoperability
  ur_exp_external_semaphore_handle_t MExternalSemaphore = nullptr;
  std::optional<uint64_t> MWaitValue;
  std::optional<uint64_t> MSignalValue;

  // The user facing node type, used for operations which are recorded to a
  // graph. Since some operations may actually be a different type than the user
  // submitted, e.g. a fill() which is performed as a kernel submission. This is
  // used to pass the type that the user expects to graph nodes when they are
  // created for later query by users.
  sycl::ext::oneapi::experimental::node_type MUserFacingNodeType =
      sycl::ext::oneapi::experimental::node_type::empty;

  // Storage for any SYCL Graph dynamic parameters which have been flagged for
  // registration in the CG, along with the argument index for the parameter.
  std::vector<std::pair<
      ext::oneapi::experimental::detail::dynamic_parameter_impl *, int>>
      MDynamicParameters;

  // Track whether an NDRange was used when submitting a kernel (as opposed to a
  // range), needed for graph update
  bool MNDRangeUsed = false;

  /// The storage for the arguments passed.
  /// We need to store a copy of values that are passed explicitly through
  /// set_arg, require and so on, because we need them to be alive after
  /// we exit the method they are passed in.
  detail::CG::StorageInitHelper CGData;

  /// The list of arguments for the kernel.
  std::vector<detail::ArgDesc> MArgs;

  /// The list of associated accessors with this handler.
  /// These accessors were created with this handler as argument or
  /// have become required for this handler via require method.
  std::vector<detail::ArgDesc> MAssociatedAccesors;

  /// Struct that encodes global size, local size, ...
  detail::NDRDescT MNDRDesc;

  /// Type of the command group, e.g. kernel, fill. Can also encode version.
  /// Use getType and setType methods to access this variable unless
  /// manipulations with version are required
  detail::CGType MCGType = detail::CGType::None;

  /// The graph that is associated with this handler.
  std::shared_ptr<ext::oneapi::experimental::detail::graph_impl> MGraph;
  /// If we are submitting a graph using ext_oneapi_graph this will be the graph
  /// to be executed.
  std::shared_ptr<ext::oneapi::experimental::detail::exec_graph_impl>
      MExecGraph;
  /// Storage for a node created from a subgraph submission.
  std::shared_ptr<ext::oneapi::experimental::detail::node_impl> MSubgraphNode;
  /// Storage for the CG created when handling graph nodes added explicitly.
  std::unique_ptr<detail::CG> MGraphNodeCG;

  /// Storage for lambda/function when using HostTask
  std::shared_ptr<detail::HostTask> MHostTask;
  /// The list of valid SYCL events that need to complete
  /// before barrier command can be executed
  std::vector<detail::EventImplPtr> MEventsWaitWithBarrier;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
