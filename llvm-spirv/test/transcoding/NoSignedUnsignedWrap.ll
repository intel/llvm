; Source
; int square(unsigned short a) {
;   return a * a;
; }
; Command
; clang -cc1 -triple spir -emit-llvm -O2 -o NoSignedUnsignedWrap.ll test.cl
;
; Positive tests:
;
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_no_integer_wrap_decoration -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_no_integer_wrap_decoration -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv --spirv-ext=+SPV_KHR_no_integer_wrap_decoration -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
;
; During consumption, any SPIR-V extension must be accepted by default
;
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
;
; Negative tests:
;
; Check that translator is able to reject SPIR-V if extension is disallowed
;
; RUN: not llvm-spirv -r %t.spv --spirv-ext=-SPV_KHR_no_integer_wrap_decoration -o - 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID-SPIRV
;
; Check that translator is able to skip nsw/nuw attributes if extension is disabled implicitly or explicitly
;
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV-NOEXT
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM-NOEXT
;
; RUN: llvm-spirv %t.bc --spirv-ext=-SPV_KHR_no_integer_wrap_decoration -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV-NOEXT
; RUN: llvm-spirv %t.bc --spirv-ext=-SPV_KHR_no_integer_wrap_decoration -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM-NOEXT

; CHECK-SPIRV: Extension "SPV_KHR_no_integer_wrap_decoration"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} NoSignedWrap
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} NoUnsignedWrap
;
; CHECK-SPIRV-NOEXT-NOT: Extension "SPV_KHR_no_integer_wrap_decoration"
; CHECK-SPIRV-NOEXT-NOT: Decorate {{[0-9]+}} NoSignedWrap
; CHECK-SPIRV-NOEXT-NOT: Decorate {{[0-9]+}} NoUnsignedWrap
;
; CHECK-INVALID-SPIRV: input SPIR-V module uses extension 'SPV_KHR_no_integer_wrap_decoration' which were disabled

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: norecurse nounwind readnone
define spir_func i32 @square(i16 zeroext %a) local_unnamed_addr #0 {
entry:
  %conv = zext i16 %a to i32
  ; CHECK-LLVM: mul nuw nsw
  ; CHECK-LLVM-NOEXT: mul i32
  %mul = mul nuw nsw i32 %conv, %conv
  ret i32 %mul
}

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
