; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_bfloat16_conversion 2>&1 \
; RUN: | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: InvalidInstruction: Can't translate llvm instruction:
; CHECK-ERROR-NEXT: ConvertBF16ToFINTEL
; CHECK-ERROR-NEXT: Input type must have the same number of components as result type


target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func void @_Z1f() {
  %1 = alloca <2 x i16>, align 4
  %2 = load <2 x i16>, <2 x i16>* %1, align 4
  %3 = tail call spir_func float @_Z27__spirv_ConvertBF16ToFINTELf(<2 x i16> %2)
  ret void
}

declare spir_func float @_Z27__spirv_ConvertBF16ToFINTELf(<2 x i16>)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 13.0.0"}
