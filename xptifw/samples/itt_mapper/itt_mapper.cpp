//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#include <chrono>
#include <mutex>
#include <stdio.h>
#include <string>
#include <thread>
#include <unordered_map>

#include "xpti/xpti_trace_framework.h"
#include "xpti_itt_helper.h"

static uint8_t g_stream_id = 0;
__itt_domain *g_stream_domain = 0;
// xpti::thread_id g_tid;
std::mutex g_io_mutex;

XPTI_CALLBACK_API void trace_point_begin(uint16_t trace_type,
                                         xpti::trace_event_data_t *parent,
                                         xpti::trace_event_data_t *event,
                                         uint64_t instance,
                                         const void *user_data);

XPTI_CALLBACK_API void trace_barrier_begin(uint16_t trace_type,
                                           xpti::trace_event_data_t *parent,
                                           xpti::trace_event_data_t *event,
                                           uint64_t instance,
                                           const void *user_data);

XPTI_CALLBACK_API void trace_wait_begin(uint16_t trace_type,
                                        xpti::trace_event_data_t *parent,
                                        xpti::trace_event_data_t *event,
                                        uint64_t instance,
                                        const void *user_data);

XPTI_CALLBACK_API void trace_point_end(uint16_t trace_type,
                                       xpti::trace_event_data_t *parent,
                                       xpti::trace_event_data_t *event,
                                       uint64_t instance,
                                       const void *user_data);

XPTI_CALLBACK_API void graph_create_handler(uint16_t trace_type,
                                            xpti::trace_event_data_t *parent,
                                            xpti::trace_event_data_t *event,
                                            uint64_t instance,
                                            const void *user_data);

XPTI_CALLBACK_API void node_create_handler(uint16_t trace_type,
                                           xpti::trace_event_data_t *parent,
                                           xpti::trace_event_data_t *event,
                                           uint64_t instance,
                                           const void *user_data);

XPTI_CALLBACK_API void edge_create_handler(uint16_t trace_type,
                                           xpti::trace_event_data_t *parent,
                                           xpti::trace_event_data_t *event,
                                           uint64_t instance,
                                           const void *user_data);

XPTI_CALLBACK_API void sycl_tp_handler(uint16_t trace_type,
                                       xpti::trace_event_data_t *parent,
                                       xpti::trace_event_data_t *event,
                                       uint64_t instance,
                                       const void *user_data);

XPTI_CALLBACK_API void xptiTraceInit(
    unsigned int major_version, ///< Major version of the runtime
    unsigned int minor_version, ///< Minor version of the runtime
    const char *version_str,    ///< Version information as a string
    const char *stream_name ///<_tbb_strings Stream name under which the stream
                            ///< from this runtime will be emitted
) {
  if (stream_name) {
    // If a new domain needs to be created for each stream, then a
    // map from stream id to domain needs to be created.
    g_stream_domain = __itt_domain_create(stream_name);
    g_stream_id = xptiRegisterStream(stream_name);

    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::graph_create,
                         graph_create_handler);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::node_create,
                         node_create_handler);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::edge_create,
                         edge_create_handler);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::region_begin,
                         trace_point_begin);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::region_end,
                         trace_point_end);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::task_begin,
                         trace_point_begin);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::task_end,
                         trace_point_end);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::barrier_begin,
                         trace_barrier_begin);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::barrier_end,
                         trace_point_end);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::lock_begin,
                         trace_point_begin);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::lock_end,
                         trace_point_end);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::transfer_begin,
                         trace_point_begin);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::transfer_end,
                         trace_point_end);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::thread_begin,
                         trace_point_begin);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::thread_end,
                         trace_point_end);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::wait_begin,
                         trace_wait_begin);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::wait_end,
                         trace_point_end);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::signal,
                         sycl_tp_handler);
  } else {
    // handle the case when a bad stream name has been provided
  }
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *stream_name) {
  // clean up any data structure/memory allocated
}

XPTI_CALLBACK_API void trace_wait_begin(uint16_t trace_type,
                                        xpti::trace_event_data_t *parent,
                                        xpti::trace_event_data_t *event,
                                        uint64_t instance,
                                        const void *user_data) {
  std::string fn;
  if (user_data) {
    fn = (const char *)user_data;
  } else {
    auto p = xptiQueryPayload(event);
    fn = p->name;
  }
  std::string name = (user_data ? (const char *)user_data : "unknown");
  if (event) {
    xpti::itt::function_begin(g_stream_domain, (void *)event->unique_id,
                              (parent ? (void *)parent->unique_id : nullptr),
                              name.c_str());
  } else {
    uint64_t task_id = xptiGetUniqueId();
    xpti::itt::function_begin(g_stream_domain, (void *)task_id,
                              (parent ? (void *)parent->unique_id : nullptr),
                              name.c_str());
  }
}

XPTI_CALLBACK_API void sycl_tp_handler(uint16_t trace_type,
                                       xpti::trace_event_data_t *parent,
                                       xpti::trace_event_data_t *event,
                                       uint64_t instance,
                                       const void *user_data) {
  if (trace_type == (uint16_t)xpti::trace_point_type_t::signal) {
    // We have our notification for now that sends a cl_event
    // as the parent
    // xpti::trace_state_t s;
    // xpti::timer::tick_t time = xpti::timer::rdtsc();
    // auto tid = xpti::timer::get_thread_id();
    // uint32_t cpu = g_tid.enum_id(tid);

    // s.event = event;
    // s.parent = parent;
    // s.time = time;
    // s.device = cpu;

    // Using the event in "user_data", you can align the OpenCL/L0 event that
    // has the kernel duration

    // std::lock_guard<std::mutex> lock(g_io_mutex);
    // g_event_map[(uint64_t)user_data] =
    // std::make_pair((uint16_t)xpti::trace_point_type_t::task_end, s);
  }
}

XPTI_CALLBACK_API void trace_barrier_begin(uint16_t trace_type,
                                           xpti::trace_event_data_t *parent,
                                           xpti::trace_event_data_t *event,
                                           uint64_t instance,
                                           const void *user_data) {
  std::string name = (user_data ? (const char *)user_data : "unknown");
  if (event) {
    xpti::itt::function_begin(g_stream_domain, (void *)event->unique_id,
                              (parent ? (void *)parent->unique_id : nullptr),
                              name.c_str());
  } else {
    uint64_t task_id = xptiGetUniqueId();
    xpti::itt::function_begin(g_stream_domain, (void *)task_id,
                              (parent ? (void *)parent->unique_id : nullptr),
                              name.c_str());
  }
}

XPTI_CALLBACK_API void trace_point_begin(uint16_t trace_type,
                                         xpti::trace_event_data_t *parent,
                                         xpti::trace_event_data_t *event,
                                         uint64_t instance,
                                         const void *user_data) {
  if (event) {
    auto p = xptiQueryPayload(event);
    if (p->name_sid() != xpti::invalid_id) {
      if (!parent)
        printf("Parent is NULL\n");
      xpti::itt::task_begin(g_stream_domain, event, parent, p);
      uint64_t task_id = xptiGetUniqueId();
      xpti::itt::function_begin(g_stream_domain, (void *)task_id,
                                (parent ? (void *)parent->unique_id : nullptr),
                                p->name);
    }
  } else {
    uint64_t task_id = xptiGetUniqueId();
    xpti::itt::function_begin(g_stream_domain, (void *)task_id,
                              (parent ? (void *)parent->unique_id : nullptr),
                              (const char *)user_data);
  }
}

XPTI_CALLBACK_API void trace_point_end(uint16_t trace_type,
                                       xpti::trace_event_data_t *parent,
                                       xpti::trace_event_data_t *event,
                                       uint64_t instance,
                                       const void *user_data) {
  auto p = xptiQueryPayload(event);
  if (p->name_sid() != xpti::invalid_id) {
    xpti::itt::task_end(g_stream_domain);
    xpti::itt::task_end(g_stream_domain);
  }
}

XPTI_CALLBACK_API void graph_create_handler(uint16_t trace_type,
                                            xpti::trace_event_data_t *parent,
                                            xpti::trace_event_data_t *event,
                                            uint64_t instance,
                                            const void *user_data) {
  auto p = xptiQueryPayload(event);
  if (p->name_sid() != xpti::invalid_id) {
    xpti::itt::create_graph(g_stream_domain, parent, event, p);
  }
}

XPTI_CALLBACK_API void node_create_handler(uint16_t trace_type,
                                           xpti::trace_event_data_t *parent,
                                           xpti::trace_event_data_t *event,
                                           uint64_t instance,
                                           const void *user_data) {
  auto p = xptiQueryPayload(event);
  if (p->name_sid() != xpti::invalid_id) {
    __itt_id node_id = xpti::itt::create_node(g_stream_domain, parent, event, p,
                                              (const char *)user_data);
    (void)node_id;
  }
}

XPTI_CALLBACK_API void edge_create_handler(uint16_t trace_type,
                                           xpti::trace_event_data_t *parent,
                                           xpti::trace_event_data_t *event,
                                           uint64_t instance,
                                           const void *user_data) {
  auto p = xptiQueryPayload(event);
  if (p->name_sid() != xpti::invalid_id) {
    xpti::itt::create_edge(g_stream_domain, parent, event);
  }
}
