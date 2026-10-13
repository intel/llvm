; RUN: llvm-spirv %s -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %s -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; An undef initializer of an LLVM array constant must not be dropped: a
; module-scope variable with internal linkage requires an initializer. It is
; translated to an OpConstantComposite of OpUndef constituents, which the
; reader folds back to undef.

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

@__const.array = private unnamed_addr addrspace(2) constant [4 x i32] undef, align 4

; Function Attrs: nounwind readnone
define spir_kernel void @k() #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  ret void
}

; CHECK-SPIRV: TypeInt [[#I32:]] 32 0
; CHECK-SPIRV: Undef [[#I32]] [[#UNDEF:]]
; CHECK-SPIRV: Constant [[#I32]] [[#LEN:]] 4
; CHECK-SPIRV: TypeArray [[#ARR:]] [[#I32]] [[#LEN]]
; CHECK-SPIRV: ConstantComposite [[#ARR]] [[#CC:]] [[#UNDEF]] [[#UNDEF]] [[#UNDEF]] [[#UNDEF]]
; CHECK-SPIRV: Variable {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[#CC]]

; CHECK-LLVM: @__const.array = internal addrspace(2) constant [4 x i32] undef, align 4

!llvm.module.flags = !{!1}
!opencl.ocl.version = !{!2}
!opencl.spir.version = !{!3}

!0 = !{}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 1, i32 0}
!3 = !{i32 1, i32 2}
