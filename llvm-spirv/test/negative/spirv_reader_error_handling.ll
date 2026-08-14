; readSpirv() is documented to report a rejected module by returning false and
; filling in an error string. It instead reaches std::exit() through
; SPIRVErrorLog::checkError(), because the error handling kind was a global
; defaulting to Exit, so the caller's error handling never ran.
;
; Check that the kind is selectable through TranslatorOpts, and that the non
; fatal kind both keeps the process alive and carries the reason out to the
; caller.
;
; RUN: echo 'not a SPIR-V module' > %t.spv
; RUN: not llvm-spirv -r --spirv-error-handling=ignore %t.spv -o %t.bc 2>&1 | FileCheck %s
;
; The reason has to survive the call, and nothing may be written to stderr
; ahead of the caller's own diagnostic.
; CHECK-NOT: {{.}}
; CHECK: Fails to load SPIR-V as LLVM Module: InvalidModule: Invalid SPIR-V module: invalid magic number
;
; The spec constant scan takes the same selection through TranslatorOpts: under
; ignore the scan reports failure through its return value and the tool prints
; its own diagnostic.
; RUN: not llvm-spirv --spec-const-info --spirv-error-handling=ignore %t.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-SPEC-IGNORE
; CHECK-SPEC-IGNORE-NOT: InvalidModule
; CHECK-SPEC-IGNORE: Invalid SPIR-V binary
;
; Under the default, exit, the process terminates inside the scan with the
; error on stderr, before the tool's own diagnostic is reached.
; RUN: not llvm-spirv --spec-const-info %t.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-SPEC-EXIT
; CHECK-SPEC-EXIT: InvalidModule: Invalid SPIR-V module: invalid magic number
; CHECK-SPEC-EXIT-NOT: Invalid SPIR-V binary
;
; The -spec-const option runs the same scan; under ignore its failure must
; still produce a diagnostic rather than a bare nonzero exit.
; RUN: not llvm-spirv -r -spec-const=0:i32:1 --spirv-error-handling=ignore \
; RUN:   %t.spv -o %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-SPECOPT-IGNORE
; CHECK-SPECOPT-IGNORE-NOT: InvalidModule
; CHECK-SPECOPT-IGNORE: Invalid SPIR-V binary
