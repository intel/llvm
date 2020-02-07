; RUN: llvm-as < %s | llvm-spirv -spirv-text -o %t
; RUN: FileCheck < %t %s

; ModuleID = 'test.cl'
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Check that only `__spirv` builtins are translated in OpenCL C++.

; Make sure that `__spirv` builtins *are* translated.
; CHECK: {{[0-9]+}} Name 5 "__spirv_BuiltInGlobalInvocationId"
declare dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_xv() #6

; Make sure that OCL builtins *are not* translated.
; CHECK: {{[0-9]+}} Name 7 "_Z13get_global_idj"
declare dso_local spir_func i64 @_Z13get_global_idj() #6

; Function Attrs: nounwind
define spir_func void @foo() #0 {
entry:
  tail call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv() #2
  tail call spir_func i64 @_Z13get_global_idj() #2
  ret void
}

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}
!llvm.ident = !{!9}
!spirv.Source = !{!10}
!spirv.String = !{!11}

!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 1}
!8 = !{}
!9 = !{!"clang version 3.6.1 "}
; 4 = OpenCL C++
!10 = !{i32 4, i32 200000, !11}
!11 = !{!"test.cl"}
