//===--------- ur_interface_loader.hpp - OpenCL Adapter ---------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <unified-runtime/ur_api.h>
#include <unified-runtime/ur_ddi.h>

namespace ur::opencl {

struct ddi_getter {
  static const ur_dditable_t *value();
};

#ifdef UR_STATIC_ADAPTER_OPENCL
ur_result_t urAdapterGetDdiTables(ur_dditable_t *ddi);
#endif

} // namespace ur::opencl
