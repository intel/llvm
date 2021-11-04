; RUN: sycl-post-link -split=kernel -emit-program-metadata -symbols -emit-exported-symbols \
; RUN:     -split-esimd -lower-esimd -O2 -spec-const=default -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefix CHECK-IR0
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefix CHECK-IR1
; RUN: FileCheck %s -input-file=%t_2.ll --check-prefix CHECK-IR2

; RUN: sycl-post-link -split=kernel -emit-program-metadata -symbols -emit-exported-symbols \
; RUN:     -split-esimd -lower-esimd -O2 -spec-const=default -reduce-memory-usage=true -S %s -o %t-red.table
; RUN: FileCheck %s -input-file=%t-red_0.ll --check-prefix CHECK-IR0
; TODO: FileCheck %s -input-file=%t-red_1.ll --check-prefix CHECK-IR1
; TODO: FileCheck %s -input-file=%t-red_2.ll --check-prefix CHECK-IR2

; This test checks that kernel info is saved for CUDA target during device code
; splitting.
; It should work for reduce-memory-usage mode as well, but now this information
; is lost for 2nd and subsequent split modules.

; CHECK-IR0: !{{[0-9]+}} = !{void (i32 addrspace(1)*)* @_ZTS5Kern1, !"kernel", i32 1}
; CHECK-IR1: !{{[0-9]+}} = !{void (i32 addrspace(1)*)* @_ZTS5Kern2, !"kernel", i32 1}
; CHECK-IR2: !{{[0-9]+}} = !{void (i32 addrspace(1)*)* @_ZTS5Kern3, !"kernel", i32 1}

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define weak_odr dso_local void @_ZTS5Kern1(i32 addrspace(1)* %_arg_) #0 {
entry:
  ret void
}

define weak_odr dso_local void @_ZTS5Kern2(i32 addrspace(1)* %_arg_) #0 {
entry:
  ret void
}

define weak_odr dso_local void @_ZTS5Kern3(i32 addrspace(1)* %_arg_) #0 {
entry:
  ret void
}

attributes #0 = { convergent mustprogress noinline norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="cuda-split-per-kernel.cpp" "target-cpu"="sm_50" "target-features"="+ptx74,+sm_50" "uniform-work-group-size"="true" }

!nvvm.annotations = !{!0, !1, !2}

!0 = !{void (i32 addrspace(1)*)* @_ZTS5Kern1, !"kernel", i32 1}
!1 = !{void (i32 addrspace(1)*)* @_ZTS5Kern2, !"kernel", i32 1}
!2 = !{void (i32 addrspace(1)*)* @_ZTS5Kern3, !"kernel", i32 1}
