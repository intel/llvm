; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -msan-poison-stack-with-call=1 -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

declare spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyiPU3AS3iPU3AS1immP13__spirv_Event(i32, ptr addrspace(3), ptr addrspace(1), i64, i64, target("spirv.Event")) nounwind
declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1aPU3AS3Kamm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, target("spirv.Event"))
declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1cPU3AS3Kcmm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, target("spirv.Event"))
declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1hPU3AS3Khmm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, target("spirv.Event"))
declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1sPU3AS3Ksmm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, target("spirv.Event"))
declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1tPU3AS3Ktmm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, target("spirv.Event"))
declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1fPU3AS3Kfmm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, target("spirv.Event"))
declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1iPU3AS3Kimm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, target("spirv.Event"))
declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1jPU3AS3Kjmm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, target("spirv.Event"))
declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1dPU3AS3Kdmm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, target("spirv.Event"))
declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1lPU3AS3Klmm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, target("spirv.Event"))
declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1mPU3AS3Kmmm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, target("spirv.Event"))
declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS3Dv4_aPU3AS1KS_mm9ocl_event(i32, ptr addrspace(3), ptr addrspace(1), i64, i64, target("spirv.Event"))

define spir_kernel void @kernel1(ptr addrspace(3) %_arg_localAcc, ptr addrspace(1) %_arg_globalAcc) sanitize_memory {
entry:
  ; CHECK-LABEL: define spir_kernel void @kernel1
  ; CHECK: @__msan_barrier()
  ; CHECK:      [[REG1:%[0-9]+]] = ptrtoint ptr addrspace(3) %_arg_localAcc to i64
  ; CHECK-NEXT: [[REG2:%[0-9]+]] = ptrtoint ptr addrspace(1) %_arg_globalAcc to i64
  ; CHECK-NEXT: call void @__msan_unpoison_strided_copy(i64 [[REG1]], i32 3, i64 [[REG2]], i32 1, i32 4, i64 512, i64 1)
  %copy = call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyiPU3AS3iPU3AS1immP13__spirv_Event(i32 2, ptr addrspace(3) %_arg_localAcc, ptr addrspace(1) %_arg_globalAcc, i64 512, i64 1, target("spirv.Event") zeroinitializer)

  ; CHECK: __msan_unpoison_strided_copy
  %copy2 = call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1aPU3AS3Kamm9ocl_event(i32 2, ptr addrspace(1) %_arg_globalAcc, ptr addrspace(3) %_arg_localAcc, i64 512, i64 1, target("spirv.Event") zeroinitializer)
  ; CHECK: __msan_unpoison_strided_copy
  %copy3 = call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1cPU3AS3Kcmm9ocl_event(i32 2, ptr addrspace(1) %_arg_globalAcc, ptr addrspace(3) %_arg_localAcc, i64 512, i64 1, target("spirv.Event") zeroinitializer)
  ; CHECK: __msan_unpoison_strided_copy
  %copy4 = call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1hPU3AS3Khmm9ocl_event(i32 2, ptr addrspace(1) %_arg_globalAcc, ptr addrspace(3) %_arg_localAcc, i64 512, i64 1, target("spirv.Event") zeroinitializer)
  ; CHECK: __msan_unpoison_strided_copy
  %copy5 = call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1sPU3AS3Ksmm9ocl_event(i32 2, ptr addrspace(1) %_arg_globalAcc, ptr addrspace(3) %_arg_localAcc, i64 512, i64 1, target("spirv.Event") zeroinitializer)
  ; CHECK: __msan_unpoison_strided_copy
  %copy6 = call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1tPU3AS3Ktmm9ocl_event(i32 2, ptr addrspace(1) %_arg_globalAcc, ptr addrspace(3) %_arg_localAcc, i64 512, i64 1, target("spirv.Event") zeroinitializer)
  ; CHECK: __msan_unpoison_strided_copy
  %copy7 = call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1fPU3AS3Kfmm9ocl_event(i32 2, ptr addrspace(1) %_arg_globalAcc, ptr addrspace(3) %_arg_localAcc, i64 512, i64 1, target("spirv.Event") zeroinitializer)
  ; CHECK: __msan_unpoison_strided_copy
  %copy8 = call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1jPU3AS3Kjmm9ocl_event(i32 2, ptr addrspace(1) %_arg_globalAcc, ptr addrspace(3) %_arg_localAcc, i64 512, i64 1, target("spirv.Event") zeroinitializer)
  ; CHECK: __msan_unpoison_strided_copy
  %copy9 = call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1dPU3AS3Kdmm9ocl_event(i32 2, ptr addrspace(1) %_arg_globalAcc, ptr addrspace(3) %_arg_localAcc, i64 512, i64 1, target("spirv.Event") zeroinitializer)
  ; CHECK: __msan_unpoison_strided_copy
  %copy10 = call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1lPU3AS3Klmm9ocl_event(i32 2, ptr addrspace(1) %_arg_globalAcc, ptr addrspace(3) %_arg_localAcc, i64 512, i64 1, target("spirv.Event") zeroinitializer)
  ; CHECK: __msan_unpoison_strided_copy
  %copy11 = call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1mPU3AS3Kmmm9ocl_event(i32 2, ptr addrspace(1) %_arg_globalAcc, ptr addrspace(3) %_arg_localAcc, i64 512, i64 1, target("spirv.Event") zeroinitializer)
  ; CHECK: __msan_unpoison_strided_copy
  %copy12 = call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS3Dv4_aPU3AS1KS_mm9ocl_event(i32 2, ptr addrspace(3) %_arg_localAcc, ptr addrspace(1) %_arg_globalAcc, i64 512, i64 1, target("spirv.Event") zeroinitializer)
  ret void
}

define spir_kernel void @kernel2(ptr addrspace(4) %tmp.ascast.i.i.i, ptr %byval-temp.i.i.i) {
entry:
  ; CHECK-LABEL: define spir_kernel void @kernel2
  ; CHECK: [[REG3:%.*]] = ptrtoint ptr addrspace(4) [[REG4:%.*]] to i64
  ; CHECK-NEXT: [[REG5:%.*]] = ptrtoint ptr [[REG6:%.*]] to i64
  ; CHECK-NEXT: call void @__msan_unpoison_copy(i64 [[REG3]], i32 4, i64 [[REG5]], i32 0, i32 1, i32 1, i64 8)
  ; CHECK-NEXT: call spir_func void @clogf(ptr addrspace(4) dead_on_unwind writable sret({ float, float }) align 4 [[REG4]], ptr noundef nonnull byval({ float, float }) align 4 [[REG6]])
  call spir_func void @clogf(ptr addrspace(4) dead_on_unwind writable sret({ float, float }) align 4 %tmp.ascast.i.i.i, ptr noundef nonnull byval({ float, float }) align 4 %byval-temp.i.i.i)
  ret void
}

define spir_kernel void @kernel3(ptr addrspace(4) %0) {
entry:
  ; CHECK-LABEL: define spir_kernel void @kernel3
  ; CHECK: [[REG7:%.*]] = ptrtoint ptr addrspace(4) [[REG8:%.*]] to i64
  ; CHECK-NEXT: [[REG9:%.*]] = ptrtoint ptr addrspace(4) [[REG10:%.*]] to i64
  ; CHECK-NEXT: call void @__msan_unpoison_copy(i64 [[REG7]], i32 4, i64 [[REG9]], i32 4, i32 4, i32 2, i64 4)
  ; CHECK-NEXT: call spir_func void @__devicelib_ConvertBF16ToFINTELVec4(ptr addrspace(4) noundef [[REG10]], ptr addrspace(4) noundef [[REG8]])
  call spir_func void @__devicelib_ConvertBF16ToFINTELVec4(ptr addrspace(4) noundef %0, ptr addrspace(4) noundef %0)
  ret void
}

declare spir_func void @clogf(ptr addrspace(4) sret({ float, float }), ptr)
declare spir_func void @__devicelib_ConvertBF16ToFINTELVec4(ptr addrspace(4), ptr addrspace(4))
