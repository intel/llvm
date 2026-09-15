; Test --spirv-target-triple that overrides the triple during reverse-translation.

; RUN: llvm-spirv %s -o %t.spv

; Default: triple derived from the (Physical64) addressing model.
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-DEFAULT
; CHECK-DEFAULT: target datalayout = "e-p:64:64:64{{.*}}-G1"
; CHECK-DEFAULT: target triple = "spir64-unknown-unknown"

; Override to an AMDGCN triple.
; RUN: llvm-spirv -r --spirv-target-triple=amdgcn-amd-amdhsa %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-AMDGCN
; CHECK-AMDGCN: target datalayout = "m:e-e-p:64:64:64{{.*}}-ni:7:8:9-p7:160:256:256:32
; CHECK-AMDGCN: target triple = "amdgcn-amd-amdhsa"

; Override to an NVPTX triple.
; RUN: llvm-spirv -r --spirv-target-triple=nvptx64-nvidia-cuda %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-NVPTX
; CHECK-NVPTX: target datalayout = "e-p:64:64:64{{.*}}-v1024:1024:1024"
; CHECK-NVPTX: target triple = "nvptx64-nvidia-cuda"

; The option only affects reverse translation; on forward translation it is
; ignored with a note.
; RUN: llvm-spirv --spirv-target-triple=amdgcn-amd-amdhsa %s -o %t.fwd.spv 2>&1 | FileCheck %s --check-prefix=CHECK-FWD
; CHECK-FWD: Note: --spirv-target-triple option ignored as it only affects translation from SPIR-V to LLVM IR

; An unknown/invalid triple override is rejected.
; RUN: not llvm-spirv -r --spirv-target-triple=not-a-triple %t.spv -o %t.err.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
; CHECK-ERR: Unknown target triple override: not-a-triple

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-unknown-unknown"

define spir_func void @f() {
entry:
  ret void
}
