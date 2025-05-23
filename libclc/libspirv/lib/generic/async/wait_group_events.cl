//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF void __spirv_GroupWaitEvents(int scope,
                                                    int num_events,
                                                    event_t *event_list) {
  __spirv_ControlBarrier(scope, Workgroup, SequentiallyConsistent);
}
