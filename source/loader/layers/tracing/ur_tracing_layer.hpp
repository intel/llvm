/*
 *
 * Copyright (C) 2023 Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_tracing_layer.h
 *
 */

#ifndef UR_TRACING_LAYER_H
#define UR_TRACING_LAYER_H 1

#include "ur_ddi.h"
#include "ur_proxy_layer.hpp"
#include "ur_util.hpp"

#define TRACING_COMP_NAME "tracing layer"

namespace tracing_layer {
///////////////////////////////////////////////////////////////////////////////
class __urdlllocal context_t : public proxy_layer_context_t {
  public:
    ur_dditable_t urDdiTable = {};

    context_t();
    ~context_t();

    bool isEnabled();
    ur_result_t init(ur_dditable_t *dditable);
    uint64_t notify_begin(uint32_t id, const char *name, void *args);
    void notify_end(uint32_t id, const char *name, void *args,
                    ur_result_t *resultp, uint64_t instance);

  private:
    void notify(uint16_t trace_type, uint32_t id, const char *name, void *args,
                ur_result_t *resultp, uint64_t instance);
    uint8_t call_stream_id;
};

extern context_t context;
} // namespace tracing_layer

#endif /* UR_TRACING_LAYER_H */
