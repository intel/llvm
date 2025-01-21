; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv -s %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_vector_compute -s %t.bc

; ModuleID = 'lower-non-standard-vec-with-ext'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@Id = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent norecurse
define dso_local spir_func void @vmult2() local_unnamed_addr #0 !sycl_explicit_simd !4 !intel_reqd_sub_group_size !6 {
entry:
  %0 = bitcast <1 x i32> <i32 65793> to <4 x i8>
  %1 = extractelement <4 x i8> %0, i32 0
  %2 = bitcast <1 x i32> <i32 131586> to <4 x i8>
  %3 = extractelement <4 x i8> %2, i32 0
  %4 = bitcast <5 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1> to <20 x i8>
  ret void
}

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="256" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="lower-external-funcs-with-z.cpp" }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!opencl.used.extensions = !{!4}
!opencl.used.optional.core.features = !{!4}
!opencl.compiler.options = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 0, i32 100000}
!4 = !{}
!5 = !{!"Compiler"}
!6 = !{i32 1}
