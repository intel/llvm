// REQUIRES: xptifw, opencl, cpu, linux
// RUN: %clangxx %s -DXPTI_COLLECTOR -DXPTI_CALLBACK_API_EXPORTS %xptifw_lib -shared -fPIC -std=c++17 -o %t_collector.so
// RUN: %{build} -o %t.out
// RUN: env UR_ENABLE_LAYERS=UR_LAYER_TRACING env XPTI_TRACE_ENABLE=1 env XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher env XPTI_SUBSCRIBERS=%t_collector.so %{run} %t.out | FileCheck %s

#include "basic_event_collection.inc"
//
// CHECK: xptiTraceInit: Stream Name = ur
// CHECK: xptiTraceInit: Stream Name = sycl.experimental.mem_alloc
// CHECK: xptiTraceInit: Stream Name = sycl
// CHECK-NEXT: Graph create
// CHECK:      UR Call Begin : urPlatformGet
// CHECK:      UR Call Begin : urContextCreate
// CHECK:      UR Call Begin : urQueueCreate
// CHECK:      UR Call Begin : urDeviceSelectBinary
// CHECK:      UR Call Begin : urKernelCreate
// CHECK-NEXT: UR Call Begin : urPlatformGetInfo
// CHECK-NEXT: UR Call Begin : urPlatformGetInfo
// CHECK-NEXT: UR Call Begin : urKernelSetExecInfo
// CHECK:      UR Call Begin : urKernelSetArgPointer
// CHECK-NEXT: UR Call Begin : urKernelGetGroupInfo
// CHECK-NEXT: UR Call Begin : urEnqueueKernelLaunch
// CHECK:      UR Call Begin : urKernelCreate
// CHECK-NEXT: UR Call Begin : urPlatformGetInfo
// CHECK-NEXT: UR Call Begin : urPlatformGetInfo
// CHECK-NEXT: UR Call Begin : urKernelSetExecInfo
// CHECK:      Node create
// CHECK-DAG:    queue_id : {{.*}}
// CHECK-DAG:    sym_line_no : {{.*}}
// CHECK-DAG:    sym_source_file_name : {{.*}}
// CHECK-DAG:    sym_function_name : typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda()#1}
// CHECK-DAG:    from_source : false
// CHECK-DAG:    kernel_name : typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda()#1}
// CHECK-DAG:    sycl_device : {{.*}}
// CHECK:      Node create
// CHECK-DAG:   queue_id : {{.*}}
// CHECK-DAG:   kernel_name : virtual_node[{{.*}}]
// CHECK-NEXT: Edge create
// CHECK-DAG:   queue_id : {{.*}}
// CHECK-DAG:   event : {{.*}}
// CHECK: Task begin
// CHECK-DAG:    queue_id : {{.*}}
// CHECK-DAG:    sym_line_no : {{.*}}
// CHECK-DAG:    sym_source_file_name : {{.*}}
// CHECK-DAG:    sym_function_name : typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda()#1}
// CHECK-DAG:    from_source : false
// CHECK-DAG:    kernel_name : typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda()#1}
// CHECK-DAG:    sycl_device : {{.*}}
// CHECK:      UR Call Begin : urKernelSetArgPointer
// CHECK-NEXT: UR Call Begin : urKernelGetGroupInfo
// CHECK-NEXT: UR Call Begin : urEnqueueKernelLaunch
// CHECK-NEXT: UR Call Begin : urKernelRelease
// CHECK-NEXT: UR Call Begin : urProgramRelease
// CHECK-NEXT: Signal
// CHECK-DAG:    queue_id : {{.*}}
// CHECK-DAG:    sym_line_no : {{.*}}
// CHECK-DAG:    sym_source_file_name : {{.*}}
// CHECK-DAG:    sym_function_name : typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda()#1}
// CHECK-DAG:    from_source : false
// CHECK-DAG:    kernel_name : typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda()#1}
// CHECK-DAG:    sycl_device : {{.*}}
// CHECK:      Task end
// CHECK-DAG:    queue_id : {{.*}}
// CHECK-DAG:    sym_line_no : {{.*}}
// CHECK-DAG:    sym_source_file_name : {{.*}}
// CHECK-DAG:    sym_function_name : typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda()#1}
// CHECK-DAG:    from_source : false
// CHECK-DAG:    kernel_name : typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda()#1}
// CHECK-DAG:    sycl_device : {{.*}}
// CHECK:      Wait begin
// CHECK-DAG:    queue_id : {{.*}}
// CHECK-NEXT: UR Call Begin : urEventWait
// CHECK-NEXT: Wait end
// CHECK-DAG:    queue_id : {{.*}}
// CHECK-NEXT: Node create
// CHECK-DAG:    queue_id : {{.*}}
// CHECK-DAG:    memory_size : {{.*}}
// CHECK-DAG:    dest_memory_ptr : {{.*}}
// CHECK-DAG:    src_memory_ptr : {{.*}}
// CHECK-DAG:    sycl_device : {{.*}}
// CHECK:      Task begin
// CHECK-DAG:    queue_id : {{.*}}
// CHECK-DAG:    memory_size : {{.*}}
// CHECK-DAG:    dest_memory_ptr : {{.*}}
// CHECK-DAG:    src_memory_ptr : {{.*}}
// CHECK-DAG:    sycl_device : {{.*}}
// CHECK:      UR Call Begin : urEnqueueUSMMemcpy
// CHECK-NEXT: Task end
// CHECK-DAG:    queue_id : {{.*}}
// CHECK-DAG:    memory_size : {{.*}}
// CHECK-DAG:    dest_memory_ptr : {{.*}}
// CHECK-DAG:    src_memory_ptr : {{.*}}
// CHECK-DAG:    sycl_device : {{.*}}
// CHECK:      UR Call Begin : urEventRelease
// CHECK-NEXT: Wait begin
// CHECK-DAG:    queue_id : {{.*}}
// CHECK-DAG:    sycl_device_type : {{.*}}
// CHECK:      UR Call Begin : urQueueFinish
// CHECK-NEXT: Wait end
// CHECK-DAG:    queue_id : {{.*}}
// CHECK-DAG:    sycl_device_type : {{.*}}
