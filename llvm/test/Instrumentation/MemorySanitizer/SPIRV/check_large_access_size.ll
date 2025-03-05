; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

; Function Attrs: sanitize_memory
define spir_kernel void @MyKernel(<3 x i32> %extractVec.i8.i.i.i) #0 {
entry:
  br label %for.body.i
for.body.i:                                       ; preds = %for.body.i, %entry
  %div.i.i.i.i.i.i = sdiv <3 x i32> zeroinitializer, %extractVec.i8.i.i.i
  br label %for.body.i
}
; CHECK-LABEL: @MyKernel
; CHECK: %0 = load <3 x i32>, ptr inttoptr (i64 ptrtoint (ptr addrspace(1) @__msan_param_tls to i64) to ptr), align 8
; CHECK: %1 = bitcast <3 x i32> %0 to i96
; CHECK: %_mscmp = icmp ne i96 %1, 0
; CHECK: call void @__msan_warning_noreturn(ptr addrspace(2) null, i32 0, ptr addrspace(2) @__msan_kernel)

attributes #0 = { sanitize_memory }
