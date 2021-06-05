;; The test serves a purpose to check if work item start/finish annotations
;; are being added by SPIRITTAnnotations pass

; RUN: opt < %s --SPIRITTAnnotations -S | FileCheck %s
; RUN: opt < %s --SPIRITTAnnotations -enable-new-pm=1 -S | FileCheck %s

; ModuleID = 'synthetic.bc'
source_filename = "synthetic.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_kernel void @_ZTSZ4mainE15kernel_function() local_unnamed_addr #0 !kernel_arg_buffer_location !4 {
entry:
; CHECK: _ZTSZ4mainE15kernel_function(
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @__itt_offload_wi_start_wrapper()
  %call.i = tail call spir_func i32 @_Z3foov() #2
  %cmp.i = icmp eq i32 %call.i, 42
  br i1 %cmp.i, label %"_ZZ4mainENK3$_0clEv.exit", label %if.end.i

if.end.i:                                         ; preds = %entry
  tail call spir_func void @_Z3boov() #2
; CHECK: call void @__itt_offload_wi_finish_wrapper()
; CHECK-NEXT: ret void
  ret void

"_ZZ4mainENK3$_0clEv.exit":                       ; preds = %entry, %if.end.i
; CHECK: call void @__itt_offload_wi_finish_wrapper()
; CHECK-NEXT: ret void
  ret void
}

; CHECK: declare void @__itt_offload_wi_start_wrapper()
; CHECK: declare void @__itt_offload_wi_finish_wrapper()

; Function Attrs: convergent
declare spir_func i32 @_Z3foov() local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func void @_Z3boov() local_unnamed_addr #1

attributes #0 = { convergent norecurse nounwind mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="../../llvm/clang/test/CodeGenSYCL/kernel-simple-instrumentation.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 13.0.0 (https://github.com/intel/llvm.git f16527331b8cd18b3e45a4a7bc13a2460c8d0d84)"}
!4 = !{}
