; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc

; On LLVM level, we'll check that the intrinsics were generated again in reverse
; translation, replacing the SPIR-V level implementations.
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM \
; RUN:   "--implicit-check-not=declare {{.*}} @spirv.llvm_umul_with_overflow_{{.*}}"

; CHECK-SPIRV: Name [[NAME_UMUL_FUNC_8:[0-9]+]] "spirv.llvm_umul_with_overflow_i8"
; CHECK-SPIRV: Name [[NAME_UMUL_FUNC_32:[0-9]+]] "spirv.llvm_umul_with_overflow_i32"
; CHECK-SPIRV: Name [[NAME_UMUL_FUNC_VEC_I64:[0-9]+]] "spirv.llvm_umul_with_overflow_v2i64"

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; CHECK-LLVM: [[UMUL_8_TY:%structtype]] = type { i8, i1 }
; CHECK-LLVM: [[UMUL_32_TY:%structtype.[0-9]+]] = type { i32, i1 }
; CHECK-LLVM: [[UMUL_VEC64_TY:%structtype.[0-9]+]] = type { <2 x i64>, <2 x i1> }

; Function Attrs: nofree nounwind writeonly
define dso_local spir_func void @_Z4foo8hhPh(i8 zeroext %a, i8 zeroext %b, i8* nocapture %c) local_unnamed_addr #0 {
entry:
  ; CHECK-LLVM: call [[UMUL_8_TY]] @llvm.umul.with.overflow.i8
  ; CHECK-SPIRV: FunctionCall [[#]] [[#]] [[NAME_UMUL_FUNC_8]]
  %umul = tail call { i8, i1 } @llvm.umul.with.overflow.i8(i8 %a, i8 %b)
  %cmp = extractvalue { i8, i1 } %umul, 1
  %umul.value = extractvalue { i8, i1 } %umul, 0
  %storemerge = select i1 %cmp, i8 0, i8 %umul.value
  store i8 %storemerge, i8* %c, align 1, !tbaa !2
  ret void
}

; CHECK-SPIRV: Function [[#]] [[NAME_UMUL_FUNC_8]]
; CHECK-SPIRV: FunctionParameter [[#]] [[VAR_A:[0-9]+]]
; CHECK-SPIRV: FunctionParameter [[#]] [[VAR_B:[0-9]+]]
; CHECK-SPIRV: IMul [[#]] [[MUL_RES:[0-9]+]] [[VAR_A]] [[VAR_B]]
; CHECK-SPIRV: UDiv [[#]] [[DIV_RES:[0-9]+]] [[MUL_RES]] [[VAR_A]]
; CHECK-SPIRV: INotEqual [[#]] [[CMP_RES:[0-9]+]] [[VAR_A]] [[DIV_RES]]
; CHECK-SPIRV: CompositeInsert [[#]] [[INSERT_RES:[0-9]+]] [[MUL_RES]]
; CHECK-SPIRV: CompositeInsert [[#]] [[INSERT_RES_1:[0-9]+]] [[CMP_RES]] [[INSERT_RES]]
; CHECK-SPIRV: ReturnValue [[INSERT_RES_1]]

; Function Attrs: nofree nounwind writeonly
define dso_local spir_func void @_Z5foo32jjPj(i32 %a, i32 %b, i32* nocapture %c) local_unnamed_addr #0 {
entry:
  ; CHECK-LLVM: call [[UMUL_32_TY]] @llvm.umul.with.overflow.i32
  ; CHECK-SPIRV: FunctionCall [[#]] [[#]] [[NAME_UMUL_FUNC_32]]
  %umul = tail call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %b, i32 %a)
  %umul.val = extractvalue { i32, i1 } %umul, 0
  %umul.ov = extractvalue { i32, i1 } %umul, 1
  %spec.select = select i1 %umul.ov, i32 0, i32 %umul.val
  store i32 %spec.select, i32* %c, align 4, !tbaa !5
  ret void
}

; Function Attrs: nofree nounwind writeonly
define dso_local spir_func void @umulo_v2i64(<2 x i64> %a, <2 x i64> %b, <2 x i64>* %p) nounwind {
  ; CHECK-LLVM: call [[UMUL_VEC64_TY]] @llvm.umul.with.overflow.v2i64
  ; CHECK-SPIRV: FunctionCall [[#]] [[#]] [[NAME_UMUL_FUNC_VEC_I64]]
  %umul = call {<2 x i64>, <2 x i1>} @llvm.umul.with.overflow.v2i64(<2 x i64> %a, <2 x i64> %b)
  %umul.val = extractvalue {<2 x i64>, <2 x i1>} %umul, 0
  %umul.ov = extractvalue {<2 x i64>, <2 x i1>} %umul, 1
  %zero = alloca <2 x i64>, align 16
  %spec.select = select <2 x i1> %umul.ov, <2 x i64> <i64 0, i64 0>, <2 x i64> %umul.val
  store <2 x i64> %spec.select, <2 x i64>* %p
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare { i8, i1 } @llvm.umul.with.overflow.i8(i8, i8) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare {<2 x i64>, <2 x i1>} @llvm.umul.with.overflow.v2i64(<2 x i64>, <2 x i64>) #1

attributes #0 = { nofree nounwind writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git ddcc7ce59150c9ebc6b0b2d61e7ef4f2525c11f4)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !3, i64 0}
