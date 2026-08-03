; Test that the datalayout emitted by the SPIR-V reader reflects ASMap and
; --spirv-function-program-addrspace.

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis \
; RUN:   | FileCheck %s --check-prefix=CHECK-DEFAULT
; RUN: llvm-spirv -r %t.spv --spirv-addrspace-map=0:4 -o - | llvm-dis \
; RUN:   | FileCheck %s --check-prefix=CHECK-PRIVATE-MAPPED
; RUN: llvm-spirv -r %t.spv --spirv-function-program-addrspace=4 -o - | llvm-dis \
; RUN:   | FileCheck %s --check-prefix=CHECK-PROG-EXPLICIT
; RUN: llvm-spirv -r %t.spv --spirv-addrspace-map=0:3 \
; RUN:   --spirv-function-program-addrspace=4 -o - | llvm-dis \
; RUN:   | FileCheck %s --check-prefix=CHECK-PROG-OVERRIDE

; CHECK-DEFAULT: target datalayout = "e-p:64:64:64-
; CHECK-DEFAULT-NOT: -A{{[0-9]}}
; CHECK-DEFAULT-NOT: -P{{[0-9]}}

; private(0)->4: -A4-P4
; CHECK-PRIVATE-MAPPED: target datalayout = "{{.*}}-A4-P4

; --spirv-function-program-addrspace=4 with no map: -P4, no -A.
; CHECK-PROG-EXPLICIT: target datalayout = "{{.*}}-P4
; CHECK-PROG-EXPLICIT-NOT: -A{{[0-9]}}

; --spirv-addrspace-map=0:3 plus explicit function AS=4: -A3-P4
; (explicit --spirv-function-program-addrspace overrides the map for -P).
; CHECK-PROG-OVERRIDE: target datalayout = "{{.*}}-A3-P4

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define spir_func void @test() {
  ret void
}
