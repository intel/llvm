//==----------- sycl_trace_collector.cpp -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file sycl_trace_collector.cpp
/// Routines to collect and print SYCL API calls.

#include "xpti/xpti_trace_framework.hpp"

#include <sycl/detail/spinlock.hpp>

#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

extern sycl::detail::SpinLock GlobalLock;

bool PrintSyclVerbose = false;

void TraceDiagnosticsMessage(xpti::trace_event_data_t * /*Parent*/,
                             xpti::trace_event_data_t *CurrentObject,
                             const char *Message) {
  if (!Message)
    return;

  std::cout << "[SYCL] Runtime reports: " << std::endl;
  std::cout << "what:  " << Message << std::endl;
  if (!CurrentObject)
    return;
  std::cout << "where: ";
  if (auto Payload = CurrentObject->reserved.payload) {
    bool HasData = false;
    if (Payload->flags & (uint64_t)xpti::payload_flag_t::SourceFileAvailable) {
      HasData = true;
      std::cout << Payload->source_file << ":" << Payload->line_no << "\t";
    }
    if (Payload->flags & (uint64_t)xpti::payload_flag_t::NameAvailable) {
      HasData = true;
      std::cout << Payload->name;
    }
    if (!HasData)
      std::cout << "No code location data is available.";
    std::cout << std::endl;
  }
}

template <typename Type>
void PrintMetadataField(
    const char *Title,
    const std::pair<xpti::string_id_t, xpti::object_id_t> &Item) {
  std::cout << "\t" << Title << ": " << xpti::getMetadata<Type>(Item).second
            << std::endl;
}

void TraceTaskExecutionSignals(xpti::trace_event_data_t * /*Parent*/,
                               xpti::trace_event_data_t *Event,
                               const void *UserData, uint64_t InstanceID,
                               bool IsBegin) {
  if (!Event)
    return;

  std::cout << "[SYCL] Task " << (IsBegin ? "begin" : "end  ")
            << " (event=" << Event << ",instanceID=" << InstanceID << ")"
            << std::endl;

  // TODO: some metadata could be added at the "end" point (e.g. memory
  // allocation and result ptr). To consider how to distinguish new data for the
  // same event appeared between begin-end points.
  if (!IsBegin || !PrintSyclVerbose)
    return;

  xpti::metadata_t *Metadata = xptiQueryMetadata(Event);
  for (const auto &Item : *Metadata) {
    std::string_view Key{xptiLookupString(Item.first)};
    if (Key == "queue_id") {
      PrintMetadataField<unsigned long long>("queue_id", Item);
    } else if (Key == "kernel_name") {
      PrintMetadataField<std::string>("kernel_name", Item);
    } else if (Key == "memory_object") {
      PrintMetadataField<void *>("memory_object", Item);
    } else if (Key == "sycl_device") {
      PrintMetadataField<void *>("sycl_device", Item);
    } else if (Key == "copy_from") {
      PrintMetadataField<void *>("copy_from", Item);
    } else if (Key == "copy_to") {
      PrintMetadataField<void *>("copy_to", Item);
    } else if (Key == "memory_size") {
      PrintMetadataField<size_t>("memory_size", Item);
    } else if (Key == "value_set") {
      PrintMetadataField<int>("value_set", Item);
    } else if (Key == "memory_ptr") {
      PrintMetadataField<void *>("memory_ptr", Item);
    } else if (Key == "src_memory_ptr") {
      PrintMetadataField<void *>("src_memory_ptr", Item);
    } else if (Key == "dest_memory_ptr") {
      PrintMetadataField<void *>("dest_memory_ptr", Item);
    } else if (Key == "sycl_device_name") {
      PrintMetadataField<std::string>("sycl_device_name", Item);
    } else if (Key == "sycl_device_type") {
      PrintMetadataField<std::string>("sycl_device_type", Item);
    } else if (Key == "offset") {
      PrintMetadataField<size_t>("offset", Item);
    } else if (Key == "access_range_start") {
      PrintMetadataField<size_t>("access_range_start", Item);
    } else if (Key == "access_range_end") {
      PrintMetadataField<size_t>("access_range_end", Item);
    } else if (Key == "allocation_type") {
      PrintMetadataField<std::string>("allocation_type", Item);
    } else if (Key == "sym_function_name") {
      PrintMetadataField<std::string>("sym_function_name", Item);
    } else if (Key == "sym_source_file_name") {
      PrintMetadataField<std::string>("sym_source_file_name", Item);
    } else if (Key == "sym_line_no") {
      PrintMetadataField<int>("sym_line_no", Item);
    } else if (Key == "sym_column_no") {
      PrintMetadataField<int>("sym_column_no", Item);
    }
  }
}

void TraceQueueLifetimeSignals(xpti::trace_event_data_t * /*Parent*/,
                               xpti::trace_event_data_t *Event,
                               const void *UserData, bool IsCreation) {
  if (!Event)
    return;

  std::cout << "[SYCL] Queue " << (IsCreation ? "create" : "destroy") << ": "
            << std::endl;
  xpti::metadata_t *Metadata = xptiQueryMetadata(Event);
  for (const auto &Item : *Metadata) {
    std::string_view Key{xptiLookupString(Item.first)};
    if (Key == "queue_id") {
      PrintMetadataField<unsigned long long>("queue_id", Item);
    } else if (Key == "queue_handle") {
      PrintMetadataField<void *>("queue_handle", Item);
    } else if (!PrintSyclVerbose || !IsCreation) {
      continue;
    } else if (Key == "sycl_device") {
      PrintMetadataField<void *>("sycl_device", Item);
    } else if (Key == "sycl_context") {
      PrintMetadataField<void *>("sycl_context", Item);
    } else if (Key == "sycl_device_name") {
      PrintMetadataField<std::string>("sycl_device_name", Item);
    } else if (Key == "is_inorder") {
      PrintMetadataField<bool>("is_inorder", Item);
    }
  }
}

XPTI_CALLBACK_API void syclCallback(uint16_t TraceType,
                                    xpti::trace_event_data_t *Parent,
                                    xpti::trace_event_data_t *Event,
                                    uint64_t InstanceID, const void *UserData) {
  std::lock_guard<sycl::detail::SpinLock> Lock{GlobalLock};
  switch (TraceType) {
  case xpti::trace_diagnostics:
    TraceDiagnosticsMessage(Parent, Event, static_cast<const char *>(UserData));
    break;
  case xpti::trace_queue_create:
    TraceQueueLifetimeSignals(Parent, Event, UserData, true);
    break;
  case xpti::trace_queue_destroy:
    TraceQueueLifetimeSignals(Parent, Event, UserData, false);
    break;
  case xpti::trace_task_begin:
    TraceTaskExecutionSignals(Parent, Event, UserData, InstanceID, true);
    break;
  case xpti::trace_task_end:
    TraceTaskExecutionSignals(Parent, Event, UserData, InstanceID, false);
    break;
  default: {
    if (PrintSyclVerbose)
      std::cout << "Trace type is unexpected. Please update trace collector."
                << std::endl;
  } break;
  }
}

void syclPrintersInit() {
  std::string_view PrinterType(std::getenv("SYCL_TRACE_PRINT_FORMAT"));
  if (PrinterType == "classic") {
    std::cerr << "Classic output is not supported yet for SYCL API\n";
  } else if (PrinterType == "verbose") {
    PrintSyclVerbose = true;
  } else if (PrinterType == "compact") {
    PrintSyclVerbose = false;
  }
}

// For unification purpose
void syclPrintersFinish() {}
