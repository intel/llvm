; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

%"struct.my_struct" = type { i8 }

@__const.struct = private unnamed_addr addrspace(2) constant %"struct.my_struct" undef, align 1

; Function Attrs: nounwind readnone
define spir_kernel void @k() #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  ret void
}

; CHECK-SPIRV: {{[0-9]+}} Undef {{[0-9]+}} [[UNDEF_ID:[0-9]+]]
; CHECK-SPIRV: {{[0-9]+}} ConstantComposite {{[0-9]+}} [[CC_ID:[0-9]+]] [[UNDEF_ID]]
; CHECK-SPIRV: {{[0-9]+}} Variable {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[CC_ID]]
; CHECK-LLVM: @__const.struct = internal addrspace(2) constant %struct.my_struct undef, align 1

!llvm.module.flags = !{!1}
!opencl.ocl.version = !{!2}
!opencl.spir.version = !{!3}

!0 = !{}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 1, i32 0}
!3 = !{i32 1, i32 2}
