//===--------- adapter.hpp - Level Zero V1 Adapter -----------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <unified-runtime/ur_api.h>

namespace ur::level_zero::v1 {

ur_result_t urAdapterGet(uint32_t NumEntries, ur_adapter_handle_t *phAdapters,
                         uint32_t *pNumAdapters);

} // namespace ur::level_zero::v1
