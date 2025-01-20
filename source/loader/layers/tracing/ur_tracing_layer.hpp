/*
 *
 * Copyright (C) 2023 Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_tracing_layer.h
 *
 */

#ifndef UR_TRACING_LAYER_H
#define UR_TRACING_LAYER_H 1

#include "logger/ur_logger.hpp"
#include "ur_ddi.h"
#include "ur_proxy_layer.hpp"
#include "ur_util.hpp"

#define TRACING_COMP_NAME "tracing layer"

namespace ur_tracing_layer {
struct XptiContextManager;

///////////////////////////////////////////////////////////////////////////////
class __urdlllocal context_t : public proxy_layer_context_t,
                               public AtomicSingleton<context_t> {
public:
  ur_dditable_t urDdiTable = {};
  codeloc_data codelocData;
  logger::Logger logger;

  context_t();
  ~context_t();

  static std::vector<std::string> getNames() { return {name}; }
  ur_result_t init(ur_dditable_t *dditable,
                   const std::set<std::string> &enabledLayerNames,
                   codeloc_data codelocData) override;
  ur_result_t tearDown() override { return UR_RESULT_SUCCESS; }
  uint64_t notify_begin(uint32_t id, const char *name, void *args);
  void notify_end(uint32_t id, const char *name, void *args,
                  ur_result_t *resultp, uint64_t instance);

private:
  void notify(uint16_t trace_type, uint32_t id, const char *name, void *args,
              ur_result_t *resultp, uint64_t instance);
  uint8_t call_stream_id;

  inline static const std::string name = "UR_LAYER_TRACING";

  std::shared_ptr<XptiContextManager> xptiContextManager;
};

context_t *getContext();
} // namespace ur_tracing_layer

#endif /* UR_TRACING_LAYER_H */
