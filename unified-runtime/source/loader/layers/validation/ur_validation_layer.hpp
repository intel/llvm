/*
 *
 * Copyright (C) 2023-2024 Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_layer.h
 *
 */
#pragma once
#include "logger/ur_logger.hpp"
#include "ur_ddi.h"
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
  logger::Logger logger;

  ur_dditable_t urDdiTable = {};

  context_t();
  ~context_t();

  static std::vector<std::string> getNames() {
    return {nameFullValidation, nameParameterValidation, nameLeakChecking,
            nameBoundsChecking, nameLifetimeValidation};
  }
  ur_result_t init(ur_dditable_t *dditable,
                   const std::set<std::string> &enabledLayerNames,
                   codeloc_data codelocData) override;
  ur_result_t tearDown() override;

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
