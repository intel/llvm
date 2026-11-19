; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_EXT_optnone -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV-EXT
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_optnone -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV-INTEL
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_EXT_optnone,+SPV_INTEL_optnone -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV-BOTH


; RUN: llvm-spirv --spirv-ext=+SPV_EXT_optnone %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; Check that optnone is correctly ignored when extension is not enabled
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV-NONE

; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM-NONE

; Note: The capability is unconditionally printed with the EXT suffix.
; CHECK-SPIRV-EXT: Capability OptNoneEXT
; CHECK-SPIRV-INTEL: Capability OptNoneEXT
; CHECK-SPIRV-BOTH: Capability OptNoneEXT

; CHECK-SPIRV-EXT: Extension "SPV_EXT_optnone"
; CHECK-SPIRV-INTEL: Extension "SPV_INTEL_optnone"
; Note: When both extensions are enabled, prefer the EXT extension.
; CHECK-SPIRV-BOTH: Extension "SPV_EXT_optnone"

; Per SPIR-V spec:
; FunctionControlDontInlineMask = 0x2 (2)
; Per SPIR-V spec extension spec:
; FunctionControlOptNoneMask = 0x10000 (65536)
; CHECK-SPIRV-EXT: Function {{[0-9]+}} {{[0-9]+}} 65538
; CHECK-SPIRV-INTEL: Function {{[0-9]+}} {{[0-9]+}} 65538
; CHECK-SPIRV-BOTH: Function {{[0-9]+}} {{[0-9]+}} 65538
; CHECK-SPIRV-NONE: Function {{[0-9]+}} {{[0-9]+}} 2

; CHECK-LLVM: define spir_func void @_Z3foov() #[[ATTR:[0-9]+]]
; CHECK-LLVM: #[[ATTR]] = { {{.*}}noinline{{.*}}optnone{{.*}} }

; CHECK-LLVM-NONE: define spir_func void @_Z3foov() #[[ATTR:[0-9]+]]
; CHECK-LLVM-NONE-NOT: #[[ATTR]] = { {{.*}}noinline{{.*}}optnone{{.*}} }

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind optnone noinline
define spir_func void @_Z3foov() #0 {
entry:
  ret void
}

attributes #0 = { nounwind optnone noinline "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 9.0.0"}
!4 = !{}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !11, i64 0}
!10 = !{!"_ZTS3bar", !11, i64 0, !7, i64 4, !12, i64 8}
!11 = !{!"int", !7, i64 0}
!12 = !{!"float", !7, i64 0}
!13 = !{!10, !7, i64 4}
!14 = !{!10, !12, i64 8}
