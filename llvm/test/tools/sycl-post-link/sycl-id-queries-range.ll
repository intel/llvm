; This test checks that the sycl-post-link tool correctly handles the
; -id-queries-range option, which adds an 'idQueriesRange' property to the
; device binary image properties.

; Default (int) mode: property should NOT be emitted
; RUN: sycl-post-link -properties -split=auto -symbols -S < %s -o %t_default.table
; RUN: FileCheck %s -input-file=%t_default_0.prop --check-prefix CHECK-DEFAULT

; RUN: sycl-post-link -properties -split=auto -symbols -S -id-queries-range=int < %s -o %t_default.table
; RUN: FileCheck %s -input-file=%t_default_0.prop --check-prefix CHECK-DEFAULT

; uint mode: property should be emitted with value 1
; RUN: sycl-post-link -properties -split=auto -symbols -S -id-queries-range=uint < %s -o %t_uint.table
; RUN: FileCheck %s -input-file=%t_uint_0.prop --check-prefix CHECK-UINT

; size_t mode: property should be emitted with value 2
; RUN: sycl-post-link -properties -split=auto -symbols -S -id-queries-range=size_t < %s -o %t_sizet.table
; RUN: FileCheck %s -input-file=%t_sizet_0.prop --check-prefix CHECK-SIZET

; By default, the 'idQueriesRange' property should not be emitted.
; CHECK-DEFAULT-NOT: idQueriesRange

; CHECK-UINT: [SYCL/misc properties]
; CHECK-UINT: idQueriesRange=1|1

; CHECK-SIZET: [SYCL/misc properties]
; CHECK-SIZET: idQueriesRange=1|2

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define dso_local spir_func noundef i32 @_Z3fooii(i32 noundef %a, i32 noundef %b) local_unnamed_addr #0 {
entry:
  %sub = sub nsw i32 %a, %b
  ret i32 %sub
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) "sycl-module-id"="test.cpp" }
