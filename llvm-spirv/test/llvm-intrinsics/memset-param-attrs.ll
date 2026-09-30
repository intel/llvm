; Verify that captures(none) on the @llvm.memset destination parameter is
; propagated to the generated spirv.llvm_memset_* helper function, and that
; FuncParamAttr NoCapture (5) is emitted for that parameter.
;
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV:     Name [[#Helper:]] "spirv.llvm_memset_p0_i32"
; CHECK-SPIRV:     Name [[#Dest:]] "dest"
; CHECK-SPIRV:     Decorate [[#Dest]] FuncParamAttr 5
; CHECK-SPIRV:     Function [[#]] [[#Helper]]
; CHECK-SPIRV:     FunctionParameter [[#]] [[#Dest]]

; CHECK-LLVM: declare void @llvm.memset.p0.i32(ptr writeonly captures(none), i8, i32, i1 immarg)

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test(ptr %dst, i8 %val, i32 %len) {
  call void @llvm.memset.p0.i32(ptr captures(none) %dst, i8 %val, i32 %len, i1 false)
  ret void
}

declare void @llvm.memset.p0.i32(ptr captures(none) writeonly, i8, i32, i1 immarg)

!opencl.ocl.version = !{!0}
!opencl.spir.version = !{!0}
!0 = !{i32 1, i32 0}
