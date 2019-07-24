// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-SPIRV: TypeDeviceEvent
// CHECK-SPIRV: 5 Function
// CHECK-SPIRV: CreateUserEvent
// CHECK-SPIRV: IsValidEvent
// CHECK-SPIRV: RetainEvent
// CHECK-SPIRV: SetUserEventStatus
// CHECK-SPIRV: CaptureEventProfilingInfo
// CHECK-SPIRV: ReleaseEvent
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @clk_event_t_test
// CHECK-LLVM: call spir_func %opencl.clk_event_t* @_Z17create_user_eventv()
// CHECK-LLVM: call spir_func i1 @_Z14is_valid_event12ocl_clkevent
// CHECK-LLVM: call spir_func void @_Z12retain_event12ocl_clkevent
// CHECK-LLVM: call spir_func void @_Z21set_user_event_status12ocl_clkeventi(%opencl.clk_event_t* %{{[a-z]+}}, i32 -42)
// CHECK-LLVM: call spir_func void @_Z28capture_event_profiling_info12ocl_clkeventiPU3AS1v(%opencl.clk_event_t* %{{[a-z]+}}, i32 1, i8 addrspace(1)* %prof)
// CHECK-LLVM: call spir_func void @_Z13release_event12ocl_clkevent
// CHECK-LLVM: ret

kernel void clk_event_t_test(global int *res, global void *prof) {
  clk_event_t e1 = create_user_event();
  *res = is_valid_event(e1);
  retain_event(e1);
  set_user_event_status(e1, -42);
  capture_event_profiling_info(e1, CLK_PROFILING_COMMAND_EXEC_TIME, prof);
  release_event(e1);
}
