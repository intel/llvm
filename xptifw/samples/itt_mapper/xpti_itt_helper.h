//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#pragma once
#include <string>
#include <unordered_map>

// From FGA for now as this is sufficient to model SYCL instrumentation
#include "itt_minimal.h"
#include "xpti/xpti_trace_framework.h"

namespace xpti {
namespace itt {

using domains_t = std::unordered_map<std::string, __itt_domain *>;
using strings_t = std::unordered_map<std::string, __itt_string_handle *>;

extern void make_id(__itt_domain *dom, __itt_id *id, void *addr,
                    unsigned long long extra);
extern __itt_domain *create_domain(const char *domain);
extern __itt_string_handle *create_string(const char *string);
extern __itt_id create_task_group(__itt_domain *domain, void *parent,
                                  unsigned long long pextra, void *group,
                                  unsigned long long gextra, const char *name);
extern __itt_id create_graph(__itt_domain *domain,
                             xpti::trace_event_data_t *parent,
                             xpti::trace_event_data_t *event,
                             const xpti::payload_t *payload);
extern __itt_id create_node(__itt_domain *domain,
                            xpti::trace_event_data_t *parent,
                            xpti::trace_event_data_t *node,
                            const xpti::payload_t *payload,
                            const char *optional_node_type);
extern void create_edge(__itt_domain *domain, xpti::trace_event_data_t *parent,
                        xpti::trace_event_data_t *edge);
extern void task_begin(__itt_domain *domain, xpti::trace_event_data_t *task,
                       xpti::trace_event_data_t *parent,
                       const xpti::payload_t *type);
extern void task_end(__itt_domain *domain);
extern void function_begin(__itt_domain *domain, void *task, void *parent,
                           const char *type);
} // namespace itt
} // namespace xpti
