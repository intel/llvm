; RUN: llvm-spirv --spirv-ext=+SPV_AMD_weak_linkage %s -o %t.spv
; RUN: llvm-spirv %t.spv --to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %s -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV-NOEXT

; CHECK-SPIRV-DAG: Capability Linkage
; CHECK-SPIRV-DAG: Capability WeakLinkageAMD
; CHECK-SPIRV: Extension "SPV_AMD_weak_linkage"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} LinkageAttributes "GV" WeakAMD
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} LinkageAttributes "square" WeakAMD

; CHECK-SPIRV-NOEXT-NOT: Capability WeakLinkageAMD
; CHECK-SPIRV-NOEXT-NOT: Extension "SPV_AMD_weak_linkage"
; CHECK-SPIRV-NOEXT-NOT: Decorate {{[0-9]+}} LinkageAttributes "GV" WeakAMD
; CHECK-SPIRV-NOEXT-NOT: Decorate {{[0-9]+}} LinkageAttributes "square" WeakAMD

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; CHECK-LLVM: @GV = weak addrspace(1) global [3 x i32] zeroinitializer, align 4
@GV = weak addrspace(1) global [3 x i32] zeroinitializer, align 4

define spir_kernel void @k() #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  %call = call spir_func i32 @square(i32 2)
  ret void
}

; CHECK-LLVM: define weak spir_func i32 @square(i32 %in)
define weak dso_local spir_func i32 @square(i32 %in) {
entry:
  %in.addr = alloca i32, align 4
  store i32 %in, ptr %in.addr, align 4
  %0 = load i32, ptr %in.addr, align 4
  %1 = load i32, ptr %in.addr, align 4
  %mul = mul nsw i32 %0, %1
  ret i32 %mul
}

!llvm.module.flags = !{!1}
!opencl.spir.version = !{!2}

!0 = !{}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 1, i32 2}
