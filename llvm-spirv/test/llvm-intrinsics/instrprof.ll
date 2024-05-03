; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

; ModuleID = 'source.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$handler = comdat any

@__profn__ = weak_odr hidden addrspace(1) constant [7 x i8] c"handler"

; CHECK-SPIRV-NOT: llvm.instrprof.increment
; CHECK-SPIRV-NOT: llvm.instrprof.increment.step
; CHECK-SPIRV-NOT: llvm.instrprof.value.profile

; CHECK-LLVM-NOT: call void @llvm.instrprof.increment
; CHECK-LLVM-NOT: declare void @llvm.instrprof.increment(ptr, i64, i32, i32)
; CHECK-LLVM-NOT: call void @llvm.instrprof.increment.step
; CHECK-LLVM-NOT: declare void @llvm.instrprof.increment.step(ptr, i64, i32, i32, i64)
; CHECK-LLVM-NOT: call void @llvm.instrprof.value.profile
; CHECK-LLVM-NOT: declare void @llvm.instrprof.value.profile(ptr, i64, i64, i32, i32)

; Function Attrs: convergent mustprogress norecurse
define weak_odr dso_local spir_kernel void @handler() #0 comdat !kernel_arg_buffer_location !1 {
entry:
  %tmp1 = alloca [7 x i8]
  call void @llvm.memcpy.p0.p1.i64(ptr %tmp1, ptr addrspace(1) @__profn__, i64 7, i1 false)
  call void @llvm.instrprof.increment(ptr %tmp1, i64 0, i32 1, i32 0)
  call void @llvm.instrprof.increment.step(ptr %tmp1, i64 0, i32 1, i32 0, i64 0)
  call void @llvm.instrprof.value.profile(ptr %tmp1, i64 0, i64 0, i32 1, i32 0)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.instrprof.increment(ptr, i64, i32, i32) #1

; Function Attrs: nounwind
declare void @llvm.instrprof.increment.step(ptr, i64, i32, i32, i64) #1

; Function Attrs: nounwind
declare void @llvm.instrprof.value.profile(ptr, i64, i64, i32, i32) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0.p1.i64(ptr nocapture, ptr addrspace(1) nocapture readonly, i64, i1) #1

attributes #0 = { convergent mustprogress norecurse }
attributes #1 = { nounwind }

!1 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
