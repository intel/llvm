; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-NOEXT
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; Verify that even though we use the fract instruction with untyped pointers enabled,
; the SPV binary is valid and we get exactly the same output IR after the reverse translation.
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_KHR_untyped_pointers
; TODO: enable back once spirv-tools are updated
; R/UN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-EXT
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

source_filename = "math_builtin_float_half.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spirv64-unknown-unknown"

; CHECK-SPIRV: TypeFloat [[HALF:[0-9]+]] 16
; CHECK-SPIRV-NOEXT: TypePointer [[HALFPTR:[0-9]+]] 7 [[HALF]]
; CHECK-SPIRV-EXT: TypeUntypedPointerKHR [[HALFPTR:[0-9]+]] 7
; CHECK-SPIRV: TypeVector [[HALFV2:[0-9]+]] [[HALF]] 2
; CHECK-SPIRV: TypePointer [[HALFV2PTR:[0-9]+]] 7 [[HALFV2]]
; CHECK-SPIRV: Constant [[HALF]] [[CONST:[0-9]+]] 14788
; CHECK-SPIRV-NOEXT: Variable [[HALFPTR]] [[ADDR:[0-9]+]] 7
; CHECK-SPIRV-EXT: UntypedVariableKHR [[HALFPTR]] [[ADDR:[0-9]+]] 7 [[HALF]]
; CHECK-SPIRV: Variable [[HALFV2PTR]] [[ADDR2:[0-9]+]] 7
; CHECK-SPIRV: ExtInst [[HALF]] [[#]] 1 fract [[CONST]] [[ADDR]]
; CHECK-SPIRV: ExtInst [[HALFV2]] [[#]] 1 fract [[#]] [[ADDR2]]

; CHECK-LLVM: %addr = alloca half
; CHECK-LLVM: %addr2 = alloca <2 x half>
; CHECK-LLVM: %res = call spir_func half @_Z5fractDhPDh(half 0xH39C4, ptr %addr)
; CHECK-LLVM: %res2 = call spir_func <2 x half> @_Z5fractDv2_DhPS_(<2 x half> <half 0xH39C4, half 0xH0000>, ptr %addr2)

define spir_kernel void @test() {
entry:
  %addr = alloca half
  %addr2 = alloca <2 x half>
  %res = call spir_func noundef half @_Z17__spirv_ocl_fractDF16_PU3AS0DF16_(half noundef 0xH39C4, ptr noundef %addr)
  %res2 = call spir_func noundef <2 x half> @_Z17__spirv_ocl_fractDv2_DF16_PU3AS0S_(<2 x half> noundef <half 0xH39C4, half 0xH0000>, ptr noundef %addr2)
  ret void
}

declare spir_func noundef half @_Z17__spirv_ocl_fractDF16_PU3AS0DF16_(half noundef, ptr noundef) local_unnamed_addr

declare spir_func noundef <2 x half> @_Z17__spirv_ocl_fractDv2_DF16_PU3AS0S_(<2 x half> noundef, ptr noundef) local_unnamed_addr
