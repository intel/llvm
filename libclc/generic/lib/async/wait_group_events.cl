#include <spirv/spirv.h>
#include <clc/clc.h>

_CLC_DEF void wait_group_events(int num_events, event_t *event_list) {
  __spirv_GroupWaitEvents(Workgroup, num_events, event_list);
}
