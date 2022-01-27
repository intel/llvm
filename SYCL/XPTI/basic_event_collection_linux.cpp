// Test is disabled to allow a few output format changes to pass pre-commit
// testing.
// REQUIRES: xptifw, opencl, cpu, linux, TEMPORARILY_DISABLED
// RUN: %clangxx %s -DXPTI_COLLECTOR -DXPTI_CALLBACK_API_EXPORTS %xptifw_lib -shared -fPIC -std=c++17 -o %t_collector.so
// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env XPTI_TRACE_ENABLE=1 env XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher env XPTI_SUBSCRIBERS=%t_collector.so %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER

#include "basic_event_collection.inc"
//
// CHECK: xptiTraceInit: Stream Name = sycl.experimental.mem_alloc
// CHECK: xptiTraceInit: Stream Name = sycl
// CHECK-NEXT: Graph create
// CHECK-NEXT: xptiTraceInit: Stream Name = sycl.pi
// CHECK-NEXT: xptiTraceInit: Stream Name = sycl.pi.debug
// CHECK:      PI Call Begin : piPlatformsGet
// CHECK:      PI Call Begin : piContextCreate
// CHECK-NEXT: PI Call Begin : piQueueCreate
// CHECK:      PI Call Begin : piextDeviceSelectBinary
// CHECK:      PI Call Begin : piKernelCreate
// CHECK-NEXT: PI Call Begin : piKernelSetExecInfo
// CHECK-NEXT: PI Call Begin : piextKernelSetArgPointer
// CHECK-NEXT: PI Call Begin : piKernelGetGroupInfo
// CHECK-NEXT: PI Call Begin : piEnqueueKernelLaunch
// CHECK-NEXT: Node create
// CHECK-NEXT:   sym_line_no : {{.*}}
// CHECK-NEXT:   sym_source_file_name : {{.*}}
// CHECK-NEXT:   sym_function_name : typeinfo name for main::{lambda(cl::sycl::handler&)#1}::operator()(cl::sycl::handler&) const::{lambda()#1}
// CHECK-NEXT:   from_source : false
// CHECK-NEXT:   kernel_name : typeinfo name for main::{lambda(cl::sycl::handler&)#1}::operator()(cl::sycl::handler&) const::{lambda()#1}
// CHECK-NEXT:   sycl_device : {{.*}}
// CHECK-NEXT: Node create
// CHECK-NEXT:   kernel_name : virtual_node[{{.*}}]
// CHECK-NEXT: Edge create
// CHECK-NEXT:   event : Event[{{.*}}]
// CHECK-NEXT: Task begin
// CHECK-NEXT:   sym_line_no : {{.*}}
// CHECK-NEXT:   sym_source_file_name : {{.*}}
// CHECK-NEXT:   sym_function_name : typeinfo name for main::{lambda(cl::sycl::handler&)#1}::operator()(cl::sycl::handler&) const::{lambda()#1}
// CHECK-NEXT:   from_source : false
// CHECK-NEXT:   kernel_name : typeinfo name for main::{lambda(cl::sycl::handler&)#1}::operator()(cl::sycl::handler&) const::{lambda()#1}
// CHECK-NEXT:   sycl_device : {{.*}}
// CHECK-NEXT: PI Call Begin : piKernelCreate
// CHECK-NEXT: PI Call Begin : piKernelSetExecInfo
// CHECK-NEXT: PI Call Begin : piextKernelSetArgPointer
// CHECK-NEXT: PI Call Begin : piKernelGetGroupInfo
// CHECK-NEXT: PI Call Begin : piEnqueueKernelLaunch
// CHECK-NEXT: Signal
// CHECK-NEXT:   sym_line_no : {{.*}}
// CHECK-NEXT:   sym_source_file_name : {{.*}}
// CHECK-NEXT:   sym_function_name : typeinfo name for main::{lambda(cl::sycl::handler&)#1}::operator()(cl::sycl::handler&) const::{lambda()#1}
// CHECK-NEXT:   from_source : false
// CHECK-NEXT:   kernel_name : typeinfo name for main::{lambda(cl::sycl::handler&)#1}::operator()(cl::sycl::handler&) const::{lambda()#1}
// CHECK-NEXT:   sycl_device : {{.*}}
// CHECK-NEXT: Task end
// CHECK-NEXT:   sym_line_no : {{.*}}
// CHECK-NEXT:   sym_source_file_name : {{.*}}
// CHECK-NEXT:   sym_function_name : typeinfo name for main::{lambda(cl::sycl::handler&)#1}::operator()(cl::sycl::handler&) const::{lambda()#1}
// CHECK-NEXT:   from_source : false
// CHECK-NEXT:   kernel_name : typeinfo name for main::{lambda(cl::sycl::handler&)#1}::operator()(cl::sycl::handler&) const::{lambda()#1}
// CHECK-NEXT:   sycl_device : {{.*}}
// CHECK-NEXT: Wait begin
// CHECK-NEXT: PI Call Begin : piEventsWait
// CHECK-NEXT: Wait end
// CHECK-NEXT: PI Call Begin : piextUSMEnqueueMemcpy
// CHECK-NEXT: PI Call Begin : piEventRelease
// CHECK-NEXT: Wait begin
// CHECK-NEXT:   sycl_device : {{.*}}
// CHECK-NEXT: PI Call Begin : piQueueFinish
// CHECK-NEXT: Wait end
// CHECK-NEXT:   sycl_device : {{.*}}
