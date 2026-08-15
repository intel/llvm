; RUN: opt -mtriple=amdgcn--amdhsa -S -passes="lto<O0>" -print-pipeline-passes %s -o - | FileCheck --check-prefix=O0 %s
; RUN: opt -mtriple=amdgcn--amdhsa -S -passes="lto<O1>" -print-pipeline-passes %s -o - | FileCheck %s
; RUN: opt -mtriple=amdgcn--amdhsa -S -passes="lto<O2>" -print-pipeline-passes %s -o - | FileCheck %s
; RUN: opt -mtriple=amdgcn--amdhsa -S -passes="lto<O3>" -print-pipeline-passes %s -o - | FileCheck %s

; RUN: opt -mtriple=amdgcn--amdhsa -S -passes="lto-pre-link<O0>" -print-pipeline-passes -amdgpu-internalize-symbols %s -o - | FileCheck --check-prefix=PRE-O0 %s
; RUN: opt -mtriple=amdgcn--amdhsa -S -passes="lto-pre-link<O1>" -print-pipeline-passes -amdgpu-internalize-symbols %s -o - | FileCheck --check-prefix=PRE %s
; RUN: opt -mtriple=amdgcn--amdhsa -S -passes="lto-pre-link<O2>" -print-pipeline-passes -amdgpu-internalize-symbols %s -o - | FileCheck --check-prefix=PRE %s
; RUN: opt -mtriple=amdgcn--amdhsa -S -passes="lto-pre-link<O3>" -print-pipeline-passes -amdgpu-internalize-symbols %s -o - | FileCheck --check-prefix=PRE %s


; CHECK: amdgpu-attributor
; O0-NOT: amdgpu-attributor

; SYCL AOT for AMDGPU runs codegen directly after the LTO pre-link pipeline
; with no post-link step, so the global offset builtin has to be resolved and
; the attributor has to run there as well.
; PRE-NOT: internalize
; PRE-NOT: printfToRuntime
; PRE: globaloffset,amdgpu-attributor

; PRE-O0-NOT: internalize
; PRE-O0-NOT: amdgpu-attributor
; PRE-O0-NOT: printfToRuntime

define amdgpu_kernel void @kernel() {
entry:
  ret void
}
