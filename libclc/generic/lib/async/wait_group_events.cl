#include <spirv/spirv.h>
#include <clc/clc.h>

_CLC_DEF void wait_group_events(int num_events, event_t *event_list) {
    _Z23__spirv_GroupWaitEventsN5__spv5ScopeEjP9ocl_event(Workgroup, num_events, event_list);
}
