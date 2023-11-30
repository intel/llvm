//===--------- virtual_mem.cpp - HIP Adapter ------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#include "context.hpp"
#include "event.hpp"
#include "physical_mem.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemGranularityGetInfo(
    ur_context_handle_t, ur_device_handle_t, ur_virtual_mem_granularity_info_t,
    size_t, void *, size_t *) {
  detail::ur::die(
      "Virtual memory extension is not currently implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemReserve(ur_context_handle_t,
                                                        const void *, size_t,
                                                        void **) {
  detail::ur::die(
      "Virtual memory extension is not currently implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemFree(ur_context_handle_t,
                                                     const void *, size_t) {
  detail::ur::die(
      "Virtual memory extension is not currently implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemSetAccess(
    ur_context_handle_t, const void *, size_t, ur_virtual_mem_access_flags_t) {
  detail::ur::die(
      "Virtual memory extension is not currently implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemMap(
    ur_context_handle_t, const void *, size_t, ur_physical_mem_handle_t, size_t,
    ur_virtual_mem_access_flags_t) {
  detail::ur::die(
      "Virtual memory extension is not currently implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemUnmap(ur_context_handle_t,
                                                      const void *, size_t) {
  detail::ur::die(
      "Virtual memory extension is not currently implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemGetInfo(ur_context_handle_t,
                                                        const void *, size_t,
                                                        ur_virtual_mem_info_t,
                                                        size_t, void *,
                                                        size_t *) {
  detail::ur::die(
      "Virtual memory extension is not currently implemented for HIP adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
