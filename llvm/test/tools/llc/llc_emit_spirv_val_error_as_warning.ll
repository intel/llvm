// REQUIRES: spirv-val,system-linux

; RUN: llc %s -filetype=obj -mtriple=spirv64-unknown-unknown -O0 --avoid-spirv-capabilities=Shader --translator-compatibility-mode --spirv-ext=+SPV_INTEL_function_pointers -o %t.spv 2>&1 | FileCheck -check-prefix=CHECK-WARNINGS %s

; Check for spirv-val warnings.

; CHECK-WARNINGS: llc: warning: SPIR-V validation started.
; CHECK-WARNINGS-DAG: error: line {{[0-9]+}}: ID '16[%_Z2f1i]' has not been defined
; CHECK-WARNINGS-DAG: llc: warning: SPIR-V validation failed.

target triple = "spirv64-unknown-unknown"

define dso_local i32 @_Z2f1i(i32 %0) {
  %2 = add nsw i32 %0, 1
  ret i32 %2
}

define dso_local i32 @_Z2f2i(i32 %0) {
  %2 = add nsw i32 %0, 2
  ret i32 %2
}

define dso_local i64 @_Z3runiiPi(i32 %0, i32 %1, ptr nocapture %2) local_unnamed_addr {
  %4 = icmp slt i32 %0, 10
  br i1 %4, label %5, label %7

5:
  %6 = add nsw i32 %1, 2
  store i32 %6, ptr %2, align 4
  br label %7

7:
  %8 = phi <2 x i64> [ <i64 ptrtoint (ptr @_Z2f2i to i64), i64 ptrtoint (ptr @_Z2f2i to i64)>, %5 ], [ <i64 ptrtoint (ptr @_Z2f2i to i64), i64 ptrtoint (ptr @_Z2f1i to i64)>, %3 ]
  %9 = extractelement <2 x i64> %8, i64 0
  %10 = extractelement <2 x i64> %8, i64 1
  %11 = add nsw i64 %9, %10
  ret i64 %11
}
