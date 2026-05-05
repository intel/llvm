/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file tsan_ddi.hpp
 *
 */

#include "unified-runtime/ur_ddi.h"

namespace ur_sanitizer_layer {

void initTsanInterceptor();
void destroyTsanInterceptor();

ur_result_t initTsanDDITable(ur_dditable_t *dditable);

} // namespace ur_sanitizer_layer
