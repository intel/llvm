;; Check that llvm.bitreverse.* intrinsics are lowered to emulation functions

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

; SPV_KHR_bit_instructions extension was not enabled so BitReverse must not be generated
; CHECK-SPIRV-NOT: BitReverse

; Check for expected bitreverse lowerings

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; scalar
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; call-sites
; CHECK-LLVM: i8 @llvm_bitreverse_i8
; CHECK-LLVM: i16 @llvm_bitreverse_i16
; CHECK-LLVM: i32 @llvm_bitreverse_i32
; CHECK-LLVM: i64 @llvm_bitreverse_i64

; definitions
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_i8
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_i16
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_i32
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_i64

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 2-element vector
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; call-sites
; CHECK-LLVM: <2 x i8> @llvm_bitreverse_v2i8
; CHECK-LLVM: <2 x i16> @llvm_bitreverse_v2i16
; CHECK-LLVM: <2 x i32> @llvm_bitreverse_v2i32
; CHECK-LLVM: <2 x i64> @llvm_bitreverse_v2i64

; definitions
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v2i8
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v2i16
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v2i32
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v2i64

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 3-element vector
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; call-sites
; CHECK-LLVM: <3 x i8> @llvm_bitreverse_v3i8
; CHECK-LLVM: <3 x i16> @llvm_bitreverse_v3i16
; CHECK-LLVM: <3 x i32> @llvm_bitreverse_v3i32
; CHECK-LLVM: <3 x i64> @llvm_bitreverse_v3i64

; definitions
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v3i8
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v3i16
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v3i32
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v3i64

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 4-element vector
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; call-sites
; CHECK-LLVM: <4 x i8> @llvm_bitreverse_v4i8
; CHECK-LLVM: <4 x i16> @llvm_bitreverse_v4i16
; CHECK-LLVM: <4 x i32> @llvm_bitreverse_v4i32
; CHECK-LLVM: <4 x i64> @llvm_bitreverse_v4i64

; definitions
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v4i8
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v4i16
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v4i32
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v4i64

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 8-element vector
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; call-sites
; CHECK-LLVM: <8 x i8> @llvm_bitreverse_v8i8
; CHECK-LLVM: <8 x i16> @llvm_bitreverse_v8i16
; CHECK-LLVM: <8 x i32> @llvm_bitreverse_v8i32
; CHECK-LLVM: <8 x i64> @llvm_bitreverse_v8i64

; definitions
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v8i8
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v8i16
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v8i32
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v8i64

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 16-element vector
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; call-sites
; CHECK-LLVM: <16 x i8> @llvm_bitreverse_v16i8
; CHECK-LLVM: <16 x i16> @llvm_bitreverse_v16i16
; CHECK-LLVM: <16 x i32> @llvm_bitreverse_v16i32
; CHECK-LLVM: <16 x i64> @llvm_bitreverse_v16i64

; definitions
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v16i8
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v16i16
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v16i32
; CHECK-LLVM: define spir_func {{.*}} @llvm_bitreverse_v16i64

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: convergent nounwind writeonly
define spir_kernel void @testBitRev(i8 %a, i16 %b, i32 %c, i64 %d, ptr addrspace(1) nocapture %res) {
entry:
  %call8 = call i8 @llvm.bitreverse.i8(i8 %a)
  store i8 %call8, ptr addrspace(1) %res, align 2, !tbaa !7

  %call16 = call i16 @llvm.bitreverse.i16(i16 %b)
  store i16 %call16, ptr addrspace(1) %res, align 2, !tbaa !7
  
  %call32 = call i32 @llvm.bitreverse.i32(i32 %c)
  store i32 %call32, ptr addrspace(1) %res, align 4, !tbaa !7

  %call64 = call i64 @llvm.bitreverse.i64(i64 %d)
  store i64 %call64, ptr addrspace(1) %res, align 8, !tbaa !7
  
  ret void
}

define spir_kernel void @testBitRevV2(<2 x i8> %a, <2 x i16> %b, <2 x i32> %c, <2 x i64> %d, ptr addrspace(1) nocapture %res) {
entry:
  %call8 = call <2 x i8> @llvm.bitreverse.v2i8(<2 x i8> %a)
  store <2 x i8> %call8, ptr addrspace(1) %res, align 2, !tbaa !7

  %call16 = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> %b)
  store <2 x i16> %call16, ptr addrspace(1) %res, align 2, !tbaa !7

  %call32 = call <2 x i32> @llvm.bitreverse.v2i32(<2 x i32> %c)
  store <2 x i32> %call32, ptr addrspace(1) %res, align 4, !tbaa !7

  %call64 = call <2 x i64> @llvm.bitreverse.v2i64(<2 x i64> %d)
  store <2 x i64> %call64, ptr addrspace(1) %res, align 8, !tbaa !7

  ret void
}

define spir_kernel void @testBitRevV3(<3 x i8> %a, <3 x i16> %b, <3 x i32> %c, <3 x i64> %d, ptr addrspace(1) nocapture %res) {
entry:
  %call8 = call <3 x i8> @llvm.bitreverse.v3i8(<3 x i8> %a)
  store <3 x i8> %call8, ptr addrspace(1) %res, align 2, !tbaa !7

  %call16 = call <3 x i16> @llvm.bitreverse.v3i16(<3 x i16> %b)
  store <3 x i16> %call16, ptr addrspace(1) %res, align 2, !tbaa !7

  %call32 = call <3 x i32> @llvm.bitreverse.v3i32(<3 x i32> %c)
  store <3 x i32> %call32, ptr addrspace(1) %res, align 4, !tbaa !7

  %call64 = call <3 x i64> @llvm.bitreverse.v3i64(<3 x i64> %d)
  store <3 x i64> %call64, ptr addrspace(1) %res, align 8, !tbaa !7

  ret void
}

define spir_kernel void @testBitRevV4(<4 x i8> %a, <4 x i16> %b, <4 x i32> %c, <4 x i64> %d, ptr addrspace(1) nocapture %res) {
entry:
  %call8 = call <4 x i8> @llvm.bitreverse.v4i8(<4 x i8> %a)
  store <4 x i8> %call8, ptr addrspace(1) %res, align 2, !tbaa !7

  %call16 = call <4 x i16> @llvm.bitreverse.v4i16(<4 x i16> %b)
  store <4 x i16> %call16, ptr addrspace(1) %res, align 2, !tbaa !7

  %call32 = call <4 x i32> @llvm.bitreverse.v4i32(<4 x i32> %c)
  store <4 x i32> %call32, ptr addrspace(1) %res, align 4, !tbaa !7

  %call64 = call <4 x i64> @llvm.bitreverse.v4i64(<4 x i64> %d)
  store <4 x i64> %call64, ptr addrspace(1) %res, align 8, !tbaa !7

  ret void
}

define spir_kernel void @testBitRevV8(<8 x i8> %a, <8 x i16> %b, <8 x i32> %c, <8 x i64> %d, ptr addrspace(1) nocapture %res) {
entry:
  %call8 = call <8 x i8> @llvm.bitreverse.v8i8(<8 x i8> %a)
  store <8 x i8> %call8, ptr addrspace(1) %res, align 2, !tbaa !7

  %call16 = call <8 x i16> @llvm.bitreverse.v8i16(<8 x i16> %b)
  store <8 x i16> %call16, ptr addrspace(1) %res, align 2, !tbaa !7

  %call32 = call <8 x i32> @llvm.bitreverse.v8i32(<8 x i32> %c)
  store <8 x i32> %call32, ptr addrspace(1) %res, align 4, !tbaa !7

  %call64 = call <8 x i64> @llvm.bitreverse.v8i64(<8 x i64> %d)
  store <8 x i64> %call64, ptr addrspace(1) %res, align 8, !tbaa !7

  ret void
}

define spir_kernel void @testBitRevV16(<16 x i8> %a, <16 x i16> %b, <16 x i32> %c, <16 x i64> %d, ptr addrspace(1) nocapture %res) {
entry:
  %call8 = call <16 x i8> @llvm.bitreverse.v16i8(<16 x i8> %a)
  store <16 x i8> %call8, ptr addrspace(1) %res, align 2, !tbaa !7

  %call16 = call <16 x i16> @llvm.bitreverse.v16i16(<16 x i16> %b)
  store <16 x i16> %call16, ptr addrspace(1) %res, align 2, !tbaa !7

  %call32 = call <16 x i32> @llvm.bitreverse.v16i32(<16 x i32> %c)
  store <16 x i32> %call32, ptr addrspace(1) %res, align 4, !tbaa !7

  %call64 = call <16 x i64> @llvm.bitreverse.v16i64(<16 x i64> %d)
  store <16 x i64> %call64, ptr addrspace(1) %res, align 8, !tbaa !7

  ret void
}

declare i8 @llvm.bitreverse.i8(i8)
declare i16 @llvm.bitreverse.i16(i16)
declare i32 @llvm.bitreverse.i32(i32)
declare i64 @llvm.bitreverse.i64(i64)

declare <2 x i8>  @llvm.bitreverse.v2i8(<2 x i8>)
declare <2 x i16> @llvm.bitreverse.v2i16(<2 x i16>)
declare <2 x i32> @llvm.bitreverse.v2i32(<2 x i32>)
declare <2 x i64> @llvm.bitreverse.v2i64(<2 x i64>)

declare <3 x i8>  @llvm.bitreverse.v3i8(<3 x i8>)
declare <3 x i16> @llvm.bitreverse.v3i16(<3 x i16>)
declare <3 x i32> @llvm.bitreverse.v3i32(<3 x i32>)
declare <3 x i64> @llvm.bitreverse.v3i64(<3 x i64>)

declare <4 x i8>  @llvm.bitreverse.v4i8(<4 x i8>)
declare <4 x i16> @llvm.bitreverse.v4i16(<4 x i16>)
declare <4 x i32> @llvm.bitreverse.v4i32(<4 x i32>)
declare <4 x i64> @llvm.bitreverse.v4i64(<4 x i64>)

declare <8 x i8>  @llvm.bitreverse.v8i8(<8 x i8>)
declare <8 x i16> @llvm.bitreverse.v8i16(<8 x i16>)
declare <8 x i32> @llvm.bitreverse.v8i32(<8 x i32>)
declare <8 x i64> @llvm.bitreverse.v8i64(<8 x i64>)

declare <16 x i8>  @llvm.bitreverse.v16i8(<16 x i8>)
declare <16 x i16> @llvm.bitreverse.v16i16(<16 x i16>)
declare <16 x i32> @llvm.bitreverse.v16i32(<16 x i32>)
declare <16 x i64> @llvm.bitreverse.v16i64(<16 x i64>)

attributes #0 = { convergent nounwind writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!3 = !{i32 0, i32 0, i32 0, i32 1}
!4 = !{!"none", !"none", !"none", !"none"}
!5 = !{!"int", !"int", !"int", !"int*"}
!6 = !{!"", !"", !"", !""}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
