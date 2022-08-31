; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc -o %t.spv 2>&1 | FileCheck %s

; CHECK: InvalidModule: Invalid SPIR-V module: Casts from private/local/global address space are allowed only to generic

source_filename = "sample.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: nofree norecurse nounwind writeonly
define dso_local spir_func void @_Z10kernel_funPi(i32 addrspace(1)* %ptr) {
entry:
  %0 = addrspacecast i32 addrspace(1)* %ptr to i32*
  ret void
}

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 12.0.0"}
!4 = !{!5, !6, i64 0}
!5 = !{!"_ZTSN2cl4sycl9multi_ptrIiLNS0_6access13address_spaceE1EEE", !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
