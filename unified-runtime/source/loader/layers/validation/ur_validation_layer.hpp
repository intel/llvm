/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_layer.h
 *
 */
#pragma once
#include "logger/ur_logger.hpp"
#include "unified-runtime/ur_ddi.h"
#include "ur_proxy_layer.hpp"
#include "ur_util.hpp"

namespace ur_validation_layer {

struct RefCountContext;

///////////////////////////////////////////////////////////////////////////////
class __urdlllocal context_t : public proxy_layer_context_t,
                               public AtomicSingleton<context_t> {
public:
  bool enableParameterValidation = false;
  bool enableBoundsChecking = false;
  bool enableLeakChecking = false;
  bool enableLifetimeValidation = false;
  bool enableLaunchBlocking = false;
  logger::Logger logger;

  ur_dditable_t urDdiTable = {};

  context_t();
  ~context_t();

  static std::vector<std::string> getNames() {
    return {nameFullValidation, nameParameterValidation, nameLeakChecking,
            nameBoundsChecking, nameLifetimeValidation,  nameLaunchBlocking};
  }
  ur_result_t init(ur_dditable_t *dditable,
                   const std::set<std::string> &enabledLayerNames,
                   codeloc_data codelocData) override;
  ur_result_t tearDown() override;

  /// Waits until every command enqueued to \p hQueue so far has completed. Never
  /// fails: a queue an adapter cannot drain stays asynchronous instead. See
  /// nameLaunchBlocking for what this cannot do.
  void blockOnQueue(ur_queue_handle_t hQueue);

  std::unique_ptr<RefCountContext> refCountContext;

private:
  inline static const std::string nameFullValidation =
      "UR_LAYER_FULL_VALIDATION";
  inline static const std::string nameParameterValidation =
      "UR_LAYER_PARAMETER_VALIDATION";
  inline static const std::string nameBoundsChecking =
      "UR_LAYER_BOUNDS_CHECKING";
  inline static const std::string nameLeakChecking = "UR_LAYER_LEAK_CHECKING";
  inline static const std::string nameLifetimeValidation =
      "UR_LAYER_LIFETIME_VALIDATION";

  /// Makes commands enqueued to a queue synchronous: an enqueue does not return
  /// until the queue has drained, so a device fault is reported where it was
  /// caused rather than at the next wait. This is what CUDA_LAUNCH_BLOCKING=1
  /// provides, and what the SYCL Runtime enables this for when
  /// SYCL_LAUNCH_BLOCKING=1 is set. It serializes the application, so it is a
  /// debugging aid and nothing else.
  ///
  /// The wait is the adapter's own queue drain, which cannot be given a
  /// deadline. An application whose enqueued work can only complete through
  /// host progress that happens after the submission returns - a kernel
  /// spinning on a host-written flag, or a barrier waiting on an interop event
  /// the application signals later - therefore hangs under this mode where it
  /// would otherwise run.
  inline static const std::string nameLaunchBlocking =
      "UR_LAYER_LAUNCH_BLOCKING";
};

ur_result_t bounds(ur_mem_handle_t buffer, size_t offset, size_t size);

ur_result_t bounds(ur_mem_handle_t buffer, ur_rect_offset_t offset,
                   ur_rect_region_t region);

ur_result_t bounds(ur_queue_handle_t queue, const void *ptr, size_t offset,
                   size_t size);

ur_result_t boundsImage(ur_mem_handle_t image, ur_rect_offset_t origin,
                        ur_rect_region_t region);

context_t *getContext();

} // namespace ur_validation_layer
