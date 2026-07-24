; An OpTypeImage with an R64ui/R64i Image Format requires the Int64ImageEXT
; capability and the SPV_EXT_shader_image_int64 extension.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_EXT_shader_image_int64
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_EXT_shader_image_int64

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV-DAG: Capability Int64ImageEXT
; CHECK-SPIRV-DAG: Extension "SPV_EXT_shader_image_int64"
; CHECK-SPIRV-DAG: TypeInt [[#Int64Ty:]] 64
; R64ui Image Format is encoded as 40, R64i as 41.
; CHECK-SPIRV-DAG: TypeImage [[#ImgUTy:]] [[#Int64Ty]] 1 0 0 0 2 40 2
; CHECK-SPIRV-DAG: TypeImage [[#ImgSTy:]] [[#Int64Ty]] 1 0 0 0 2 41 2

; CHECK-LLVM-DAG: target("spirv.Image", i64, 1, 0, 0, 0, 2, 40, 2)
; CHECK-LLVM-DAG: target("spirv.Image", i64, 1, 0, 0, 0, 2, 41, 2)

define spir_func void @foo(target("spirv.Image", i64, 1, 0, 0, 0, 2, 40, 2) %img) {
  ret void
}

define spir_func void @bar(target("spirv.Image", i64, 1, 0, 0, 0, 2, 41, 2) %img) {
  ret void
}

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 17.0.0"}
