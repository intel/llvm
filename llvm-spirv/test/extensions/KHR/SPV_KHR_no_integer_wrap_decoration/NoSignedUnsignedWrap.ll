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
; RUN: llvm-spirv %t.bc --spirv-max-version=1.1 --spirv-ext=+SPV_KHR_no_integer_wrap_decoration -spirv-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-EXT
; RUN: llvm-spirv %t.bc --spirv-max-version=1.1 --spirv-ext=+SPV_KHR_no_integer_wrap_decoration -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv --spirv-max-version=1.1 --spirv-ext=+SPV_KHR_no_integer_wrap_decoration -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv %t.bc --spirv-max-version=1.4 -spirv-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-NOEXT
; RUN: llvm-spirv %t.bc --spirv-max-version=1.4 -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv --spirv-max-version=1.4 -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
;
; During consumption, any SPIR-V extension must be accepted by default
;
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
;
; Negative tests:
;
; Check that translator is able to skip nsw/nuw attributes if extension is
; disabled implicitly or explicitly and if max SPIR-V version is lower then 1.4
;
; RUN: llvm-spirv %t.bc --spirv-max-version=1.1 -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV-NEGATIVE
; RUN: llvm-spirv --spirv-max-version=1.1 %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM-NEGATIVE
;
; RUN: llvm-spirv %t.bc --spirv-max-version=1.1 --spirv-ext=-SPV_KHR_no_integer_wrap_decoration -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV-NEGATIVE
; RUN: llvm-spirv %t.bc --spirv-max-version=1.1 --spirv-ext=-SPV_KHR_no_integer_wrap_decoration -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM-NEGATIVE

; Check SPIR-V versions in a format magic number + version
; CHECK-SPIRV-EXT: 119734787 65536
; CHECK-SPIRV-EXT: Extension "SPV_KHR_no_integer_wrap_decoration"
; CHECK-SPIRV-NOEXT: 119734787 66560
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} NoSignedWrap
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} NoUnsignedWrap
;
; CHECK-SPIRV-NEGATIVE-NOT: Extension "SPV_KHR_no_integer_wrap_decoration"
; CHECK-SPIRV-NEGATIVE-NOT: Decorate {{[0-9]+}} NoSignedWrap
; CHECK-SPIRV-NEGATIVE-NOT: Decorate {{[0-9]+}} NoUnsignedWrap
;
; CHECK-INVALID-SPIRV: input SPIR-V module uses extension 'SPV_KHR_no_integer_wrap_decoration' which were disabled

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: norecurse nounwind readnone
define spir_func i32 @square(i16 zeroext %a) local_unnamed_addr #0 {
entry:
  %conv = zext i16 %a to i32
  ; CHECK-LLVM: mul nuw nsw
  ; CHECK-LLVM-NEGATIVE: mul i32
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
