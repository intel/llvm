//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <stdio.h>
#include <string>
#include <thread>
#include <unordered_map>

#include "xpti_timers.hpp"
#include "xpti_trace_framework.h"

static uint8_t g_stream_id = 0;
std::mutex g_io_mutex;
xpti::thread_id g_tid;

static const char *tp_types[] = {
    "unknown", "graph_create", "node_create", "edge_create",
    "region_", "task_",        "barrier_",    "lock_",
    "signal ", "transfer_",    "thread_",     "wait_",
    0};

// The lone callback function we are going to use to demonstrate how to attach
// the collector to the running executable
XPTI_CALLBACK_API void tp_callback(uint16_t trace_type,
                                   xpti::trace_event_data_t *parent,
                                   xpti::trace_event_data_t *event,
                                   uint64_t instance, const void *user_data);

// Based on the documentation, every subscriber MUST implement the
// xptiTraceInit() and xptiTraceFinish() APIs for their subscriber collector to
// be loaded successfully.
XPTI_CALLBACK_API void xptiTraceInit(unsigned int major_version,
                                     unsigned int minor_version,
                                     const char *version_str,
                                     const char *stream_name) {
  // The basic collector will take in streams from anyone as we are just
  // printing out the stream data
  if (stream_name) {
    char *tstr;
    // Register this stream to get the stream ID; This stream may already have
    // been registered by the framework and will return the previously
    // registered stream ID
    g_stream_id = xptiRegisterStream(stream_name);
    xpti::string_id_t dev_id = xptiRegisterString("sycl_device", &tstr);

    // Register our lone callback to all pre-defined trace point types
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::graph_create,
                         tp_callback);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::node_create,
                         tp_callback);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::edge_create,
                         tp_callback);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::region_begin,
                         tp_callback);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::region_end,
                         tp_callback);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::task_begin,
                         tp_callback);
    xptiRegisterCallback(
        g_stream_id, (uint16_t)xpti::trace_point_type_t::task_end, tp_callback);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::barrier_begin,
                         tp_callback);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::barrier_end,
                         tp_callback);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::lock_begin,
                         tp_callback);
    xptiRegisterCallback(
        g_stream_id, (uint16_t)xpti::trace_point_type_t::lock_end, tp_callback);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::transfer_begin,
                         tp_callback);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::transfer_end,
                         tp_callback);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::thread_begin,
                         tp_callback);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::thread_end,
                         tp_callback);
    xptiRegisterCallback(g_stream_id,
                         (uint16_t)xpti::trace_point_type_t::wait_begin,
                         tp_callback);
    xptiRegisterCallback(
        g_stream_id, (uint16_t)xpti::trace_point_type_t::wait_end, tp_callback);
    xptiRegisterCallback(
        g_stream_id, (uint16_t)xpti::trace_point_type_t::signal, tp_callback);
    printf("Registered all callbacks\n");
  } else {
    // handle the case when a bad stream name has been provided
    std::cerr << "Invalid stream - no callbacks registered!\n";
  }
}

//
std::string truncate(std::string name) {
  size_t pos = name.find_last_of(":");
  if (pos != std::string::npos) {
    return name.substr(pos + 1);
  } else {
    return name;
  }
}

const char *extract_value(xpti::trace_event_data_t *event) {
  auto data = xptiQueryMetadata(event);
  char *str_ptr;
  xpti::string_id_t kernel_id = xptiRegisterString("kernel_name", &str_ptr);
  xpti::string_id_t memory_id = xptiRegisterString("memory_object", &str_ptr);
  if (data->count(kernel_id)) {
    return xptiLookupString((*data)[kernel_id]);
  } else if (data->count(memory_id)) {
    return xptiLookupString((*data)[memory_id]);
  }
  return event->reserved.payload->name;
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *stream_name) {
  // We do nothing here
}

XPTI_CALLBACK_API void tp_callback(uint16_t trace_type,
                                   xpti::trace_event_data_t *parent,
                                   xpti::trace_event_data_t *event,
                                   uint64_t instance, const void *user_data) {
  auto p = xptiQueryPayload(event);
  xpti::timer::tick_t time = xpti::timer::rdtsc();
  auto tid = xpti::timer::get_thread_id();
  uint32_t cpu = g_tid.enum_id(tid);
  std::string name;

  if (p->name_sid != xpti::invalid_id) {
    name = truncate(p->name);
  } else {
    name = "<unknown>";
  }

  uint64_t id = event ? event->unique_id : 0;
  // Lock while we print information
  std::lock_guard<std::mutex> lock(g_io_mutex);
  // Print the record information
  printf("%-25lu: name=%-35s cpu=%3d event_id=%10lu\n", time, name.c_str(), cpu,
         id);
  // Go through all available meta-data for an event and print it out
  xpti::metadata_t *metadata = xptiQueryMetadata(event);
  for (auto &item : *metadata) {
    printf("   %-25s:%s\n", xptiLookupString(item.first),
           xptiLookupString(item.second));
  }

  if (p->source_file_sid != xpti::invalid_id && p->line_no > 0) {
    printf("---[Source file:line no] %s:%d\n", p->source_file, p->line_no);
  }
}

#if (defined(_WIN32) || defined(_WIN64))

#include <string>
#include <windows.h>

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fwdReason, LPVOID lpvReserved) {
  switch (fwdReason) {
  case DLL_PROCESS_ATTACH:
    // printf("Framework initialization\n");
    break;
  case DLL_PROCESS_DETACH:
    //
    //  We cannot unload all subscribers here...
    //
    // printf("Framework finalization\n");
    break;
  }

  return TRUE;
}

#else // Linux (possibly macOS?)

__attribute__((constructor)) static void framework_init() {
  // printf("Framework initialization\n");
}

__attribute__((destructor)) static void framework_fini() {
  // printf("Framework finalization\n");
}

#endif