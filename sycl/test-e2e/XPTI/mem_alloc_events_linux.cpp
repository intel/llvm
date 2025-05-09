// REQUIRES: xptifw, level_zero, gpu, linux
// RUN: %build_collector
// RUN: %{build} -o %t.out
// RUN: env XPTI_TRACE_ENABLE=1 env XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher env XPTI_SUBSCRIBERS=%t_collector.dll %{run} %t.out | FileCheck %s

#include "mem_alloc_events.inc"

// CHECK:      xptiTraceInit: Stream Name = sycl.experimental.mem_alloc
// CHECK:      Mem Alloc Begin : mem_obj_handle:{{.*}}|alloc_pointer:0x0|alloc_size:400
// CHECK:      Mem Alloc End : mem_obj_handle:{{.*}}|alloc_pointer:{{.*}}|alloc_size:400
// CHECK:      Mem Release Begin : mem_obj_handle:{{.*}}|alloc_pointer:{{.*}}|alloc_size:0
// CHECK:      Mem Release End :   mem_obj_handle:{{.*}}|alloc_pointer:{{.*}}|alloc_size:0
