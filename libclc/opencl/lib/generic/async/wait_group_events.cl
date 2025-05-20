//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/clc.h>
#include <libspirv/spirv.h>

_CLC_DEF _CLC_OVERLOAD void wait_group_events(int num_events,
                                              event_t *event_list){ 
  __spirv_GroupWaitEvents(Workgroup, num_events, event_list);
}
