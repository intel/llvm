; This test verifies the behavior of the sycl-post-link tool without the -properties and -symbols options.
; In particular, we verify that the properties and symbols files are not added to the output table.
;
; RUN: sycl-post-link -split=source -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table -check-prefix=CHECK-TABLE

; CHECK-TABLE: [Code]
; CHECK-TABLE: {{.*}}_0.ll
; CHECK-TABLE: {{.*}}_1.ll

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-FOO: define dso_local spir_func noundef void @foo
define dso_local spir_func noundef void @foo(i32 noundef %a, i32 noundef %b) local_unnamed_addr #0 {
entry:
ret void
}

define dso_local spir_func noundef void @bar(i32 noundef %a, i32 noundef %b) #1 {
entry:
ret void
}
attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) "sycl-module-id"="test.cpp" "sycl-grf-size"="128" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind "sycl-module-id"="test.cpp" "sycl-grf-size"="256" }
