; This test verifies the -embed-aux-info-as-metadata option.
; In particular, we should see the properties and symbols file are not added to the output table
; and are instead embedded in the module as metadata.
;
; RUN: sycl-post-link -split=source -embed-aux-info-as-metadata -symbols -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table -check-prefix=CHECK-TABLE
; RUN: FileCheck %s -input-file=%t_0.ll -check-prefix=CHECK-BAR
; RUN: FileCheck %s -input-file=%t_1.ll -check-prefix=CHECK-FOO

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
; CHECK-FOO: !sycl_properties = !{![[#FOO_PROP:]]}
; CHECK-FOO: !sycl_symbol_table = !{![[#FOO_SYM:]]}
; CHECK-FOO: ![[#FOO_PROP]] = !{!"[SYCL/devicelib req mask]\0ADeviceLibReqMask=1|0\0A[SYCL/device requirements]\0Aaspects=2|AAAAAAAAAAA\0A[SYCL/misc properties]\0Asycl-grf-size=1|128\0A"}
; CHECK-FOO: ![[#FOO_SYM]] = !{!"foo\0A"}

; CHECK-BAR: !sycl_properties = !{![[#BAR_PROP:]]}
; CHECK-BAR: !sycl_symbol_table = !{![[#BAR_SYM:]]}
; CHECK-BAR: ![[#BAR_PROP]] = !{!"[SYCL/devicelib req mask]\0ADeviceLibReqMask=1|0\0A[SYCL/device requirements]\0Aaspects=2|AAAAAAAAAAA\0A[SYCL/misc properties]\0Asycl-grf-size=1|256\0A"}
; CHECK-BAR: ![[#BAR_SYM]] = !{!"bar\0A"}
