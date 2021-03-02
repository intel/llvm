//==--- itt_stubs.cpp - stub functions for ITT  ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device_itt.h"

#ifdef __SPIR__

DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES void
__itt_spirv_wi_start_stub(size_t *group_id, size_t wi_id, uint32_t wg_size) {}

DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES void
__itt_spirv_wi_finish_stub(size_t *group_id, size_t wi_id) {}

DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES void
__itt_spirv_wg_barrier_stub(uintptr_t barrier_id) {}

DEVICE_EXTERN_C ITT_STUB_ATTRIBUTES void
__itt_spirv_wi_resume_stub(size_t *group_id, size_t wi_id) {}

#endif // __SPIR__
