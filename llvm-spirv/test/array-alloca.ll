; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv

; Validation test.
; RUN: spirv-val %t.spv

; SPIR-V codegen test.
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; Roundtrip test.
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_KHR_untyped_pointers
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_KHR_untyped_pointers -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Decorate [[#ARR:]] Alignment 4
; CHECK-SPIRV: Decorate [[#BITARR:]] Alignment 4

; CHECK-SPIRV: TypeInt [[#I32:]] 32 0
; CHECK-SPIRV: TypeInt [[#I64:]] 64 0
; CHECK-SPIRV: Constant [[#I64]] [[#SIZE:]] 4 0
; CHECK-SPIRV: TypeVoid [[#VOID:]]
; CHECK-SPIRV: TypeFunction [[#FUNCTY:]] [[#VOID]]
; CHECK-SPIRV: {{(TypePointer|TypeUntypedPointerKHR)}} [[#PTRTY:]] [[#FUNCSTORAGE:]]
; CHECK-SPIRV: TypeArray [[#ARRTY:]] [[#I32]] [[#SIZE]]
; CHECK-SPIRV: TypePointer [[#ARRPTRTY:]] [[#FUNCSTORAGE]] [[#ARRTY]]

; CHECK-SPIRV: Function [[#VOID]] {{.*}} [[#FUNCTY]]
; CHECK-SPIRV: Variable [[#ARRPTRTY]] [[#ARR]] [[#FUNCSTORAGE]]
; CHECK-SPIRV: Bitcast [[#PTRTY]] [[#BITARR]] [[#ARR]]

; Generated LLVM is different, but equivalent as an array type is allocated.
; CHECK-LLVM:  define spir_func void @test_array_alloca()
; CHECK-LLVM:    %[[#ALLOC:]] = alloca [4 x i32], align 4
; CHECK-LLVM:    %{{.*}} = bitcast ptr %[[#ALLOC]] to ptr
define dso_local void @test_array_alloca() #0 {
  %arr = alloca i32, i64 4, align 4
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!1}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!2}

!1 = !{i32 1, i32 2}
!2 = !{}
!3 = !{}
