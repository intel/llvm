// REQUIRES: xptifw, opencl
// RUN: %clangxx %s -DXPTI_COLLECTOR -DXPTI_CALLBACK_API_EXPORTS %xptifw_lib %shared_lib %fPIC %cxx_std_optionc++17 -o %t_collector.dll
// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env XPTI_TRACE_ENABLE=1 env XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher env XPTI_SUBSCRIBERS=%t_collector.dll env SYCL_DEVICE_FILTER=opencl %t.out | FileCheck %s 2>&1

#ifdef XPTI_COLLECTOR

#include "Inputs/test_collector.cpp"

#else

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q{sycl::default_selector{}};

  auto Ptr = sycl::malloc_device<int>(1, Q);

  auto Evt1 = Q.single_task([=]() { Ptr[0] = 1; });

  auto Evt2 = Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Evt1);
    CGH.single_task([=]() { Ptr[0]++; });
  });

  Evt2.wait();

  int Res = 0;
  Q.memcpy(&Res, Ptr, 1);
  Q.wait();

  assert(Res == 2);

  return 0;
}

#endif

// CHECK: xptiTraceInit: Stream Name = sycl.experimental.mem_alloc
// CHECK: xptiTraceInit: Stream Name = sycl
// CHECK-NEXT: Graph create
// CHECK-NEXT: xptiTraceInit: Stream Name = sycl.pi
// CHECK-NEXT: xptiTraceInit: Stream Name = sycl.pi.debug
// CHECK-NEXT: PI Call Begin : piPlatformsGet
// CHECK-NEXT: PI Call Begin : piPlatformsGet
// CHECK-NEXT: PI Call Begin : piDevicesGet
// CHECK-NEXT: PI Call Begin : piDevicesGet
// CHECK-NEXT: PI Call Begin : piDeviceGetInfo
// CHECK-NEXT: PI Call Begin : piDeviceGetInfo
// CHECK-NEXT: PI Call Begin : piDeviceGetInfo
// CHECK-NEXT: PI Call Begin : piDeviceRetain
// CHECK-NEXT: PI Call Begin : piDeviceGetInfo
// CHECK-NEXT: PI Call Begin : piDeviceGetInfo
// CHECK-NEXT: PI Call Begin : piPlatformGetInfo
// CHECK-NEXT: PI Call Begin : piPlatformGetInfo
// CHECK-NEXT: PI Call Begin : piDeviceRelease
// CHECK:      PI Call Begin : piContextCreate
// CHECK-NEXT: PI Call Begin : piQueueCreate
// CHECK-NEXT: PI Call Begin : piextUSMDeviceAlloc
// CHECK-NEXT: PI Call Begin : piextDeviceSelectBinary
// CHECK-NEXT: PI Call Begin : piDeviceGetInfo
// CHECK:      PI Call Begin : piKernelCreate
// CHECK-NEXT: PI Call Begin : piKernelSetExecInfo
// CHECK-NEXT: PI Call Begin : piextKernelSetArgPointer
// CHECK-NEXT: PI Call Begin : piKernelGetGroupInfo
// CHECK-NEXT: PI Call Begin : piEnqueueKernelLaunch
// CHECK-NEXT: Node create
// CHECK-NEXT:   sym_line_no : 21
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
// CHECK-NEXT:   sym_line_no : 21
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
// CHECK-NEXT:   sym_line_no : 21
// CHECK-NEXT:   sym_source_file_name : {{.*}}
// CHECK-NEXT:   sym_function_name : typeinfo name for main::{lambda(cl::sycl::handler&)#1}::operator()(cl::sycl::handler&) const::{lambda()#1}
// CHECK-NEXT:   from_source : false
// CHECK-NEXT:   kernel_name : typeinfo name for main::{lambda(cl::sycl::handler&)#1}::operator()(cl::sycl::handler&) const::{lambda()#1}
// CHECK-NEXT:   sycl_device : {{.*}}
// CHECK-NEXT: Task end
// CHECK-NEXT:   sym_line_no : 21
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
// CHECK-NEXT: PI Call Begin : piEventRelease
// CHECK-NEXT: PI Call Begin : piEventRelease
// CHECK-NEXT: PI Call Begin : piQueueRelease
// CHECK-NEXT: PI Call Begin : piContextRelease
// CHECK-NEXT: PI Call Begin : piKernelRelease
// CHECK-NEXT: PI Call Begin : piKernelRelease
// CHECK-NEXT: PI Call Begin : piProgramRelease
// CHECK-NEXT: PI Call Begin : piDeviceRelease
// CHECK-NEXT: PI Call Begin : piTearDown
