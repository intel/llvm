//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: Stop manually mangling this name. Need C++ namespaces to get the exact mangling.
_CLC_DEF void _Z23__spirv_GroupWaitEventsN5__spv5ScopeEjP9ocl_event(
    enum Scope scope, int num_events, event_t *event_list);
