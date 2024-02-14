; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_subgroup_requirements -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %t.bc -o %t2.spv
; RUN: llvm-spirv %t2.spv -to-text -o %t2.spt
; RUN: FileCheck < %t2.spt %s --check-prefix=CHECK-SPIRV-2

; RUN: llvm-spirv -r %t2.spv -o %t2.rev.bc
; RUN: llvm-dis < %t2.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability SubgroupRequirementsINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_subgroup_requirements"
; CHECK-SPIRV: EntryPoint 6 [[kernel:[0-9]+]] "_ZTSZ4mainE7Kernel1"
; CHECK-SPIRV: ExecutionMode [[kernel]] 6446 0

; CHECK-LLVM: spir_kernel void @_ZTSZ4mainE7Kernel1() {{.*}} !intel_reqd_sub_group_size ![[MD:[0-9]+]]
; CHECK-LLVM: ![[MD]] = !{i32 0}

; CHECK-SPIRV-2-NOT: Capability SubgroupRequirementsINTEL
; CHECK-SPIRV-2-NOT: Extension "SPV_INTEL_subgroup_requirements"
; CHECK-SPIRV-2: EntryPoint 6 [[kernel:[0-9]+]] "_ZTSZ4mainE7Kernel1"
; CHECK-SPIRV-2: ExecutionMode [[kernel]] 35 0

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZTSZ4mainE7Kernel1 = comdat any

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZ4mainE7Kernel1() local_unnamed_addr #0 comdat !srcloc !5 !kernel_arg_buffer_location !6 !sycl_fixed_targets !6 !sycl_kernel_omit_args !6 !intel_reqd_sub_group_size !7 {
entry:
  ret void
}

attributes #0 = { mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="ex.cpp" "sycl-optlevel"="2" "sycl-sub-group-size"="0" "uniform-work-group-size"="true" }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 18.0.0git (/ws/llvm/clang 8fd29b3c2aa9f9ce163be557b51de39c95aaf230)"}
!5 = !{i32 358}
!6 = !{}
!7 = !{i32 0}
