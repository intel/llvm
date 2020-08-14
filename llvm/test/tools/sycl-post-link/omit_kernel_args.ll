; This test checks that the post-link tool generates correct kernel parameter
; optimization info into a property file if the source IR contained
; corresponding metadata.
;
; RUN: sycl-post-link -emit-param-info -S %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files.table --check-prefixes CHECK-TABLE
; RUN: FileCheck %s -input-file=%t.files_0.prop --match-full-lines --check-prefixes CHECK-PROP

target triple = "spir64-unknown-unknown-sycldevice"

define weak_odr spir_kernel void @SpirKernel1(float %arg1) !spir_kernel_omit_args !0 {
  call void @foo(float %arg1)
  ret void
}

define weak_odr spir_kernel void @SpirKernel2(i8 %arg1, i8 %arg2, i8 %arg3) !spir_kernel_omit_args !1 {
  call void @bar(i8 %arg1)
  call void @bar(i8 %arg2)
  call void @bar(i8 %arg3)
  ret void
}

declare void @foo(float)
declare void @bar(i8)

; // HEX: 0x01 (1 byte)
; // BIN: 01 - 2 arguments total, 1 remains
; // NOTE: In the metadata string below, unlike bitmask above, least significant
; //       element goes first
!0 = !{i1 true, i1 false}

; // HEX: 0x7E 0x06 (2 bytes)
; // BIN: 110 01111110 - 11 arguments total, 3 remain
!1 = !{i1 false, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 true, i1 true}

; CHECK-PROP: [SYCL/kernel param opt]
; // Base64 encoding in the prop file (including 8 bytes length):
; CHECK-PROP-NEXT-DAG: SpirKernel1=2|CAAAAAAAAAQA
; // Base64 encoding in the prop file (including 8 bytes length):
; CHECK-PROP-NEXT-DAG: SpirKernel2=2|LAAAAAAAAAgfGA

; CHECK-TABLE: [Code|Properties]
; CHECK-TABLE-NEXT: {{.*}}files_0.prop
; CHECK-TABLE-EMPTY:
