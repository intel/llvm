; This test checks that functions which are neither SYCL external functions nor
; part of any call graph, but have their address taken, are retained in split
; modules.

; -- Per-source split
; RUN: sycl-post-link -properties -split=source -emit-param-info -symbols -emit-exported-symbols -split-esimd -lower-esimd -O2 -spec-const=native -S < %s -o %tA.table
; RUN: FileCheck %s -input-file=%tA_0.ll --check-prefixes CHECK-A0
; RUN: FileCheck %s -input-file=%tA_1.ll --check-prefixes CHECK-A1
; -- No split
; RUN: sycl-post-link -properties -emit-param-info -symbols -emit-exported-symbols -split-esimd -lower-esimd -O2 -spec-const=native -S < %s -o %tB.table
; RUN: FileCheck %s -input-file=%tB_0.ll --check-prefixes CHECK-B0
; -- Per-kernel split
; RUN: sycl-post-link -properties -split=kernel -emit-param-info -symbols -emit-exported-symbols -split-esimd -lower-esimd -O2 -spec-const=native -S < %s -o %tC.table
; RUN: FileCheck %s -input-file=%tC_0.ll --check-prefixes CHECK-C0
; RUN: FileCheck %s -input-file=%tC_1.ll --check-prefixes CHECK-C1


; ModuleID = 'in.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$foo = comdat any
$bar = comdat any

@"tableX" = weak global [1 x void ()*] [void ()* @foo], align 8
@"tableY" = weak global [1 x void ()*] [void ()* @bar], align 8


; Function Attrs: mustprogress norecurse nounwind
define linkonce_odr dso_local spir_func void @foo() unnamed_addr #0 comdat align 2 {
; CHECK-A0: define linkonce_odr dso_local spir_func void @foo
; CHECK-A1: define linkonce_odr dso_local spir_func void @foo
; CHECK-B0: define linkonce_odr dso_local spir_func void @foo
; CHECK-B1: define linkonce_odr dso_local spir_func void @foo
; CHECK-C0: define linkonce_odr dso_local spir_func void @foo
; CHECK-C1: define linkonce_odr dso_local spir_func void @foo
  ret void
}

; -- Also check that function called from an addr-taken function is also added
;    to every split module.
; Function Attrs: mustprogress norecurse nounwind
define weak dso_local spir_func void @baz() #3 {
; CHECK-A0: define weak dso_local spir_func void @baz
; CHECK-A1: define weak dso_local spir_func void @baz
; CHECK-B0: define weak dso_local spir_func void @baz
; CHECK-B1: define weak dso_local spir_func void @baz
; CHECK-C0: define weak dso_local spir_func void @baz
; CHECK-C1: define weak dso_local spir_func void @baz
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define linkonce_odr dso_local spir_func void @bar() unnamed_addr #1 comdat align 2 {
; CHECK-A0: define linkonce_odr dso_local spir_func void @bar
; CHECK-A1: define linkonce_odr dso_local spir_func void @bar
; CHECK-B0: define linkonce_odr dso_local spir_func void @bar
; CHECK-B1: define linkonce_odr dso_local spir_func void @bar
; CHECK-C0: define linkonce_odr dso_local spir_func void @bar
; CHECK-C1: define linkonce_odr dso_local spir_func void @bar
  call void @baz()
  ret void
}

define weak_odr dso_local spir_kernel void @Kernel1() #2 {
  ret void
}

define weak_odr dso_local spir_kernel void @Kernel2() #3 {
  ret void
}

attributes #0 = { mustprogress norecurse nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "vector_function_ptrs"="tableX()" }
attributes #1 = { mustprogress norecurse nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "vector_function_ptrs"="tableY()" }
attributes #2 = { "sycl-module-id"="module1.cpp" "uniform-work-group-size"="true" }
attributes #3 = { "sycl-module-id"="module2.cpp" "uniform-work-group-size"="true" }


!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{}
!3 = !{!"<ID>"}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!7, !8, i64 8}
!7 = !{!"_ZTS4Base", !8, i64 8}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
