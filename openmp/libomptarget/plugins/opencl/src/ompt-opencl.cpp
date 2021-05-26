#if INTEL_COLLAB
//===--- OMPT support -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/cl.h>
#include "omptargetplugin.h"
#include "omptarget-tools.h"

const char *OmptDocument = nullptr;
OmptGlobalTy *OmptGlobal = nullptr;

extern int DebugLevel;
#define IDP(...)                                                               \
  do {                                                                         \
    if (DebugLevel > 0) {                                                      \
      fprintf(stderr, "Target OPENCL RTL --> ");                               \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (false)

///
/// OMPT entries for this device
///

/// Device extensions
#define FOREACH_OMPT_DEVICE_ENTRIES(Fn)                                        \
  Fn(ompt_get_device_num_procs)                                                \
  Fn(ompt_get_device_time)                                                     \
  Fn(ompt_translate_time)                                                      \
  Fn(ompt_set_trace_ompt)                                                      \
  Fn(ompt_set_trace_native)                                                    \
  Fn(ompt_start_trace)                                                         \
  Fn(ompt_pause_trace)                                                         \
  Fn(ompt_flush_trace)                                                         \
  Fn(ompt_stop_trace)                                                          \
  Fn(ompt_advance_buffer_cursor)                                               \
  Fn(ompt_get_record_type)                                                     \
  Fn(ompt_get_record_ompt)                                                     \
  Fn(ompt_get_record_native)                                                   \
  Fn(ompt_get_record_abstract)                                                 \
  Fn(ompt_ext_get_num_teams)                                                   \
  Fn(ompt_ext_get_thread_limit)                                                \
  Fn(ompt_ext_get_code_location)

/// Return available OpenMP PROCS on the device.
static int ompt_get_device_num_procs_fn(ompt_device_t *device) {
  cl_uint maxComputeUnits;
  cl_int rc = clGetDeviceInfo((cl_device_id)device,
                              CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
                              &maxComputeUnits, nullptr);
  return (rc == CL_SUCCESS) ? (int)maxComputeUnits : 0;
}

/// Return the current time on the device.
static ompt_device_time_t ompt_get_device_time_fn(ompt_device_t *device) {
  cl_ulong deviceTime, hostTime;
  cl_int rc = clGetDeviceAndHostTimer((cl_device_id)device, &deviceTime,
                                      &hostTime);
  return (rc == CL_SUCCESS) ? (ompt_device_time_t)deviceTime : 0;
}

/// Return the current walltime translated from the given device time.
static double ompt_translate_time_fn(ompt_device_t *device,
                                     ompt_device_time_t deviceTime) {
  // TODO
  IDP("Warning: ompt_translate_time_t entry is not available\n");
  return 0.0;
}

/// Enable/disable tracing of the specified OMPT events.
static ompt_set_result_t ompt_set_trace_ompt_fn(ompt_device_t *device,
                                                unsigned int enable,
                                                unsigned int etype) {
  // TODO
  IDP("Warning: ompt_set_trace_ompt_t entry is not available\n");
  return ompt_set_never;
}

/// Enable/disable tracing of the specified native events.
static ompt_set_result_t ompt_set_trace_native_fn(ompt_device_t *device,
                                                  int enable, int flags) {
  // TODO
  IDP("Warning: ompt_set_trace_native_t entry is not available\n");
  return ompt_set_never;
}

/// Start tracing on the device.
static int ompt_start_trace_fn(ompt_device_t *device,
                               ompt_callback_buffer_request_t request,
                               ompt_callback_buffer_complete_t complete) {
  // TODO
  IDP("Warning: ompt_start_trace_t entry is not available\n");
  return OMPT_FAIL;
}

/// Pause or restart tracing on the device.
static int ompt_pause_trace_fn(ompt_device_t *device, int beginPause) {
  // TODO
  IDP("Warning: ompt_pause_trace_t entry is not available\n");
  return OMPT_FAIL;
}

/// Cause the implementation issue buffer completion callbacks for the all
/// uncompleted buffers.
static int ompt_flush_trace_fn(ompt_device_t *device) {
  // TODO
  IDP("Warning: ompt_flush_trace_t entry is not available\n");
  return OMPT_FAIL;
}

/// Stop tracing on the device.
static int ompt_stop_trace_fn(ompt_device_t *device) {
  // TODO
  IDP("Warning: ompt_stop_trace_t entry is not available\n");
  return OMPT_FAIL;
}

/// Advance a trace buffer cursor to the next record.
static int ompt_advance_buffer_cursor_fn(
    ompt_device_t *device, ompt_buffer_t *buffer, size_t size,
    ompt_buffer_cursor_t current, ompt_buffer_cursor_t *next) {
  // TODO
  IDP("Warning: ompt_advance_buffer_cursor_t entry is not available\n");
  return OMPT_FAIL;
}

/// Return the type of the record at the specified cursor.
static ompt_record_t ompt_get_record_type_fn(ompt_buffer_t *buffer,
                                             ompt_buffer_cursor_t current) {
  // TODO
  IDP("Warning: ompt_get_record_type_t entry is not available\n");
  return ompt_record_invalid;
}

/// Return a pointer to an OMPT trace record at the cursor.
static ompt_record_ompt_t *ompt_get_record_ompt_fn(
    ompt_buffer_t *buffer, ompt_buffer_cursor_t current) {
  // TODO
  IDP("Warning: ompt_get_record_ompt_t entry is not available\n");
  return nullptr;
}

/// Return a pointer to a native record at the cursor.
static void *ompt_get_record_native_fn(ompt_buffer_t *buffer,
                                       ompt_buffer_cursor_t current,
                                       ompt_id_t *hostOpId) {
  // TODO
  IDP("Warning: ompt_get_record_native_t entry is not available\n");
  return nullptr;
}

/// Return an abstract record from the native record.
static ompt_record_abstract_t *ompt_get_record_abstract_fn(void *record) {
  // TODO
  IDP("Warning: ompt_get_record_abstract_t entry is not available\n");
  return nullptr;
}


///
/// Non-standard extensions
///

/// Return the number of assigned teams for the given target ID.
static int ompt_ext_get_num_teams_fn(ompt_id_t targetId) {
  if (targetId != OmptGlobal->getTrace().TargetId) {
    IDP("Warning: cannot find num_teams for target %" PRIu64 "\n", targetId);
    return 0;
  }
  return OmptGlobal->getTrace().NumTeams;
}

/// Return the number of assigned threads for the given target ID.
static int ompt_ext_get_thread_limit_fn(ompt_id_t targetId) {
  if (targetId != OmptGlobal->getTrace().TargetId) {
    IDP("Warning: cannot find thread_limit for target %" PRIu64 "\n", targetId);
    return 0;
  }
  return OmptGlobal->getTrace().ThreadLimit;
}

/// Return the code location string associated with the return address.
static const char *ompt_ext_get_code_location_fn(const void *returnAddress) {
  return OmptGlobal->getTrace().getCodeLocation(returnAddress);
}

/// Let libomptarget initialize OMPT global data, and plugin obtains access to
/// the global data and initializes document in this routine.
EXTERN void __tgt_rtl_init_ompt(void *omptGlobal) {
  OmptGlobal = (OmptGlobalTy *)omptGlobal;
  if (!OmptGlobal) {
    IDP("Warning: cannot initialize OMPT\n");
    return;
  }
  OmptDocument =
      "\nOMPT entry: \"ompt_get_device_num_procs\""
      "\n  typedef int (*ompt_get_device_num_procs_t)("
      "\n      ompt_device_t *device"
      "\n  );"
      "\n  Returns available number of processors on the device."
      "\n"
      "\nOMPT entry: \"ompt_get_device_time\""
      "\n  typedef ompt_device_time_t (*ompt_get_device_time_t)("
      "\n      ompt_device_t *device"
      "\n  );"
      "\n  Returns the current timestamp on the device."
      "\n"
      "\nOMPT entry extension: \"ompt_ext_get_num_teams\""
      "\n  typedef int (*ompt_ext_get_num_teams_t)("
      "\n      ompt_id_t target_id"
      "\n  );"
      "\n  Returns the number of teams assigned to the target region."
      "\n"
      "\nOMPT entry extension: \"ompt_ext_get_thread_limit\""
      "\n  typedef int (*ompt_ext_get_thread_limit_t)("
      "\n      ompt_id_t target_id"
      "\n  );"
      "\n  Returns the number of threads assigned to the target region."
      "\n"
      "\nOMPT entry extension: \"ompt_ext_get_code_location\""
      "\n  typedef const char *(*ompt_ext_get_code_location_t)("
      "\n      const void *codeptr_ra"
      "\n  );"
      "\n  Returns the code location string for the specified return address."
      "\n  Returns null if a code location is not found."
      "\n";
  IDP("Initialized OMPT\n");
}

ompt_interface_fn_t omptLookupEntries(const char *name) {
#define LOOKUP(Entry)                                                          \
  Entry##_t Entry##_ptr = Entry##_fn;                                          \
  if (std::strcmp(name, #Entry) == 0)                                          \
    return (ompt_interface_fn_t)Entry##_ptr;
  FOREACH_OMPT_DEVICE_ENTRIES(LOOKUP)
#undef LOOKUP
  return (ompt_interface_fn_t)0;
}

#endif // INTEL_COLLAB
