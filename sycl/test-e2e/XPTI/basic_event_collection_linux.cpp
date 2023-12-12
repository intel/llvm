// Test is disabled to allow a few output format changes to pass pre-commit
// testing.
// REQUIRES: xptifw, opencl, cpu, linux
// RUN: %clangxx %s -DXPTI_COLLECTOR -DXPTI_CALLBACK_API_EXPORTS %xptifw_lib -shared -fPIC -std=c++17 -o %t_collector.so
// RUN: %{build} -o %t.out
// RUN: env XPTI_TRACE_ENABLE=1 env XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher env XPTI_SUBSCRIBERS=%t_collector.so %{run} %t.out | FileCheck %s

#include "basic_event_collection.inc"
//
// CHECK: xptiTraceInit: Stream Name = sycl.experimental.mem_alloc
// CHECK: xptiTraceInit: Stream Name = sycl
// CHECK-NEXT: Graph create
// CHECK-NEXT: xptiTraceInit: Stream Name = sycl.pi
// CHECK-NEXT: xptiTraceInit: Stream Name = sycl.pi.debug
// CHECK:      PI Call Begin : piPlatformsGet
// CHECK:      PI Call Begin : piContextCreate
// CHECK:      PI Call Begin : piextQueueCreate
// CHECK:      PI Call Begin : piextDeviceSelectBinary
// CHECK:      PI Call Begin : piKernelCreate
// CHECK-NEXT: PI Call Begin : piKernelSetExecInfo
// CHECK:      PI Call Begin : piextKernelSetArgPointer
// CHECK-NEXT: PI Call Begin : piKernelGetGroupInfo
// CHECK-NEXT: PI Call Begin : piEnqueueKernelLaunch
// CHECK:      PI Call Begin : piKernelCreate
// CHECK-NEXT: PI Call Begin : piKernelSetExecInfo
// CHECK:      Node create
// CHECK-DAG:    sym_line_no : {{.*}}
// CHECK-DAG:    sym_source_file_name : {{.*}}
// CHECK-DAG:    sym_function_name : typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda()#1}
// CHECK-DAG:    from_source : false
// CHECK-DAG:    kernel_name : typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda()#1}
// CHECK-DAG:    sycl_device : {{.*}}
// CHECK-NEXT: Node create
// CHECK-NEXT:   kernel_name : virtual_node[{{.*}}]
// CHECK-NEXT: Edge create
// CHECK-NEXT:   event : {{.*}}
// CHECK-NEXT: Task begin
// CHECK-DAG:    sym_line_no : {{.*}}
// CHECK-DAG:    sym_source_file_name : {{.*}}
// CHECK-DAG:    sym_function_name : typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda()#1}
// CHECK-DAG:    from_source : false
// CHECK-DAG:    kernel_name : typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda()#1}
// CHECK-DAG:    sycl_device : {{.*}}
// CHECK:      PI Call Begin : piextKernelSetArgPointer
// CHECK-NEXT: PI Call Begin : piKernelGetGroupInfo
// CHECK-NEXT: PI Call Begin : piEnqueueKernelLaunch
// CHECK-NEXT: PI Call Begin : piKernelRelease
// CHECK-NEXT: PI Call Begin : piProgramRelease
// CHECK-NEXT: Signal
// CHECK-DAG:    sym_line_no : {{.*}}
// CHECK-DAG:    sym_source_file_name : {{.*}}
// CHECK-DAG:    sym_function_name : typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda()#1}
// CHECK-DAG:    from_source : false
// CHECK-DAG:    kernel_name : typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda()#1}
// CHECK-DAG:    sycl_device : {{.*}}
// CHECK-NEXT: Task end
// CHECK-DAG:    sym_line_no : {{.*}}
// CHECK-DAG:    sym_source_file_name : {{.*}}
// CHECK-DAG:    sym_function_name : typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda()#1}
// CHECK-DAG:    from_source : false
// CHECK-DAG:    kernel_name : typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda()#1}
// CHECK-DAG:    sycl_device : {{.*}}
// CHECK-NEXT: Wait begin
// CHECK-NEXT: PI Call Begin : piEventsWait
// CHECK-NEXT: Wait end
// CHECK-NEXT: Node create
// CHECK-DAG:    memory_size : {{.*}}
// CHECK-DAG:    dest_memory_ptr : {{.*}}
// CHECK-DAG:    src_memory_ptr : {{.*}}
// CHECK-DAG:    sycl_device : {{.*}}
// CHECK-NEXT: Task begin
// CHECK-DAG:    memory_size : {{.*}}
// CHECK-DAG:    dest_memory_ptr : {{.*}}
// CHECK-DAG:    src_memory_ptr : {{.*}}
// CHECK-DAG:    sycl_device : {{.*}}
// CHECK-NEXT: PI Call Begin : piextUSMEnqueueMemcpy
// CHECK-NEXT: Task end
// CHECK-DAG:    memory_size : {{.*}}
// CHECK-DAG:    dest_memory_ptr : {{.*}}
// CHECK-DAG:    src_memory_ptr : {{.*}}
// CHECK-DAG:    sycl_device : {{.*}}
// CHECK-NEXT: PI Call Begin : piEventRelease
// CHECK-NEXT: Wait begin
// CHECK:        sycl_device_type : {{.*}}
// CHECK:      PI Call Begin : piQueueFinish
// CHECK-NEXT: Wait end
// CHECK:        sycl_device_type : {{.*}}
