//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_DEF void _Z23__spirv_GroupWaitEventsjiP9ocl_event(unsigned int scope,
                                                    int num_events,
                                                    event_t __attribute__((address_space(0)))* event_list) {
  __spirv_ControlBarrier(scope, Workgroup, SequentiallyConsistent);
}

_CLC_OVERLOAD _CLC_DEF void __spirv_GroupWaitEvents(unsigned int scope,
                                                    int num_events,
                                                    event_t *event_list) {
  __spirv_ControlBarrier(scope, Workgroup, SequentiallyConsistent);
}

