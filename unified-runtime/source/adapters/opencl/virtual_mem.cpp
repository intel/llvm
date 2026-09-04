//===--------- virtual_mem.cpp - OpenCL Adapter ---------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#include "context.hpp"
#include "physical_mem.hpp"

namespace ur::opencl {

ur_result_t urVirtualMemGranularityGetInfo(ur_context_handle_t,
                                           ur_device_handle_t, size_t,
                                           ur_virtual_mem_granularity_info_t,
                                           size_t, void *, size_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urVirtualMemReserve(ur_context_handle_t, const void *, size_t,
                                void **) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urVirtualMemFree(ur_context_handle_t, const void *, size_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urVirtualMemSetAccess(ur_context_handle_t, const void *, size_t,
                                  ur_virtual_mem_access_flags_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urVirtualMemMap(ur_context_handle_t, const void *, size_t,
                            ur_physical_mem_handle_t, size_t,
                            ur_virtual_mem_access_flags_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urVirtualMemUnmap(ur_context_handle_t, const void *, size_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urVirtualMemGetInfo(ur_context_handle_t, const void *, size_t,
                                ur_virtual_mem_info_t, size_t, void *,
                                size_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

} // namespace ur::opencl
