; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Name [[NAME_FSHR_FUNC_32:[0-9]+]] "spirv.llvm_fshr_i32"
; CHECK-SPIRV: Name [[NAME_FSHR_FUNC_16:[0-9]+]] "spirv.llvm_fshr_i16"
; CHECK-SPIRV: Name [[NAME_FSHR_FUNC_VEC_INT_16:[0-9]+]] "spirv.llvm_fshr_v2i16"
; CHECK-SPIRV: TypeInt [[TYPE_INT_32:[0-9]+]] 32 0
; CHECK-SPIRV: TypeInt [[TYPE_INT_16:[0-9]+]] 16 0
; CHECK-SPIRV-DAG: Constant [[TYPE_INT_32]] [[CONST_ROTATE_32:[0-9]+]] 8
; CHECK-SPIRV-DAG: Constant [[TYPE_INT_16]] [[CONST_ROTATE_16:[0-9]+]] 8
; CHECK-SPIRV-DAG: Constant [[TYPE_INT_32]] [[CONST_TYPE_SIZE_32:[0-9]+]] 32
; CHECK-SPIRV: TypeFunction [[TYPE_ORIG_FUNC_32:[0-9]+]] [[TYPE_INT_32]] [[TYPE_INT_32]] [[TYPE_INT_32]]
; CHECK-SPIRV: TypeFunction [[TYPE_FSHR_FUNC_32:[0-9]+]] [[TYPE_INT_32]] [[TYPE_INT_32]] [[TYPE_INT_32]] [[TYPE_INT_32]]
; CHECK-SPIRV: TypeFunction [[TYPE_ORIG_FUNC_16:[0-9]+]] [[TYPE_INT_16]] [[TYPE_INT_16]] [[TYPE_INT_16]]
; CHECK-SPIRV: TypeFunction [[TYPE_FSHR_FUNC_16:[0-9]+]] [[TYPE_INT_16]] [[TYPE_INT_16]] [[TYPE_INT_16]] [[TYPE_INT_16]]
; CHECK-SPIRV: TypeVector [[TYPE_VEC_INT_16:[0-9]+]] [[TYPE_INT_16]] 2
; CHECK-SPIRV: TypeFunction [[TYPE_ORIG_FUNC_VEC_INT_16:[0-9]+]] [[TYPE_VEC_INT_16]] [[TYPE_VEC_INT_16]] [[TYPE_VEC_INT_16]]
; CHECK-SPIRV: TypeFunction [[TYPE_FSHR_FUNC_VEC_INT_16:[0-9]+]] [[TYPE_VEC_INT_16]] [[TYPE_VEC_INT_16]] [[TYPE_VEC_INT_16]] [[TYPE_VEC_INT_16]]
; CHECK-SPIRV: ConstantComposite [[TYPE_VEC_INT_16]] [[CONST_ROTATE_VEC_INT_16:[0-9]+]] [[CONST_ROTATE_16]] [[CONST_ROTATE_16]]

; On LLVM level, we'll check that the intrinsics were generated again in reverse translation,
; replacing the SPIR-V level implementations.
; CHECK-LLVM-NOT: declare {{.*}} @spirv.llvm_fshr_{{.*}}

; Function Attrs: nounwind readnone
; CHECK-SPIRV: Function [[TYPE_INT_32]] {{[0-9]+}} {{[0-9]+}} [[TYPE_ORIG_FUNC_32]]
; CHECK-SPIRV: FunctionParameter [[TYPE_INT_32]] [[X:[0-9]+]]
; CHECK-SPIRV: FunctionParameter [[TYPE_INT_32]] [[Y:[0-9]+]]
define spir_func i32 @Test_i32(i32 %x, i32 %y) local_unnamed_addr #0 {
entry:
  ; CHECK-SPIRV: FunctionCall [[TYPE_INT_32]] [[CALL_32_X_Y:[0-9]+]] [[NAME_FSHR_FUNC_32]] [[X]] [[Y]] [[CONST_ROTATE_32]]
  ; CHECK-LLVM: call i32 @llvm.fshr.i32
  %0 = call i32 @llvm.fshr.i32(i32 %x, i32 %y, i32 8)
  ; CHECK-SPIRV: FunctionCall [[TYPE_INT_32]] [[CALL_32_Y_X:[0-9]+]] [[NAME_FSHR_FUNC_32]] [[Y]] [[X]] [[CONST_ROTATE_32]]
  ; CHECK-LLVM: call i32 @llvm.fshr.i32
  %1 = call i32 @llvm.fshr.i32(i32 %y, i32 %x, i32 8)
  ; CHECK-SPIRV: IAdd [[TYPE_INT_32]] [[ADD_32:[0-9]+]] [[CALL_32_X_Y]] [[CALL_32_Y_X]]
  %sum = add i32 %0, %1
  ; CHECK-SPIRV: ReturnValue [[ADD_32]]
  ret i32 %sum
}

; CHECK-SPIRV: Function [[TYPE_INT_32]] [[NAME_FSHR_FUNC_32]] {{[0-9]+}} [[TYPE_FSHR_FUNC_32]]
; CHECK-SPIRV: FunctionParameter [[TYPE_INT_32]] [[X_ARG:[0-9]+]]
; CHECK-SPIRV: FunctionParameter [[TYPE_INT_32]] [[Y_ARG:[0-9]+]]
; CHECK-SPIRV: FunctionParameter [[TYPE_INT_32]] [[ROT:[0-9]+]]

; CHECK-SPIRV: UMod [[TYPE_INT_32]] [[ROTATE_MOD_SIZE:[0-9]+]] [[ROT]] [[CONST_TYPE_SIZE_32]]
; CHECK-SPIRV: ShiftRightLogical [[TYPE_INT_32]] [[Y_SHIFT_RIGHT:[0-9]+]] [[Y_ARG]] [[ROTATE_MOD_SIZE]]
; CHECK-SPIRV: ISub [[TYPE_INT_32]] [[NEG_ROTATE:[0-9]+]] [[CONST_TYPE_SIZE_32]] [[ROTATE_MOD_SIZE]]
; CHECK-SPIRV: ShiftLeftLogical [[TYPE_INT_32]] [[X_SHIFT_LEFT:[0-9]+]] [[X_ARG]] [[NEG_ROTATE]]
; CHECK-SPIRV: BitwiseOr [[TYPE_INT_32]] [[FSHR_RESULT:[0-9]+]] [[Y_SHIFT_RIGHT]] [[X_SHIFT_LEFT]]
; CHECK-SPIRV: ReturnValue [[FSHR_RESULT]]

; Function Attrs: nounwind readnone
; CHECK-SPIRV: Function [[TYPE_INT_16]] {{[0-9]+}} {{[0-9]+}} [[TYPE_ORIG_FUNC_16]]
; CHECK-SPIRV: FunctionParameter [[TYPE_INT_16]] [[X:[0-9]+]]
; CHECK-SPIRV: FunctionParameter [[TYPE_INT_16]] [[Y:[0-9]+]]
define spir_func i16 @Test_i16(i16 %x, i16 %y) local_unnamed_addr #0 {
entry:
  ; CHECK-SPIRV: FunctionCall [[TYPE_INT_16]] [[CALL_16:[0-9]+]] [[NAME_FSHR_FUNC_16]] [[X]] [[Y]] [[CONST_ROTATE_16]]
  ; CHECK-LLVM: call i16 @llvm.fshr.i16
  %0 = call i16 @llvm.fshr.i16(i16 %x, i16 %y, i16 8)
  ; CHECK-SPIRV: ReturnValue [[CALL_16]]
  ret i16 %0
}

; Just check that the function for i16 was generated as such - we've checked the logic for another type.
; CHECK-SPIRV: Function [[TYPE_INT_16]] [[NAME_FSHR_FUNC_16]] {{[0-9]+}} [[TYPE_FSHR_FUNC_16]]
; CHECK-SPIRV: FunctionParameter [[TYPE_INT_16]] [[X_ARG:[0-9]+]]
; CHECK-SPIRV: FunctionParameter [[TYPE_INT_16]] [[Y_ARG:[0-9]+]]
; CHECK-SPIRV: FunctionParameter [[TYPE_INT_16]] [[ROT:[0-9]+]]

; CHECK-SPIRV: Function [[TYPE_VEC_INT_16]] {{[0-9]+}} {{[0-9]+}} [[TYPE_ORIG_FUNC_VEC_INT_16]]
; CHECK-SPIRV: FunctionParameter [[TYPE_VEC_INT_16]] [[X:[0-9]+]]
; CHECK-SPIRV: FunctionParameter [[TYPE_VEC_INT_16]] [[Y:[0-9]+]]
define spir_func <2 x i16> @Test_v2i16(<2 x i16> %x, <2 x i16> %y) local_unnamed_addr #0 {
entry:
  ; CHECK-SPIRV: FunctionCall [[TYPE_VEC_INT_16]] [[CALL_VEC_INT_16:[0-9]+]] [[NAME_FSHR_FUNC_VEC_INT_16]] [[X]] [[Y]] [[CONST_ROTATE_VEC_INT_16]]
  ; CHECK-LLVM: call <2 x i16> @llvm.fshr.v2i16
  %0 = call <2 x i16> @llvm.fshr.v2i16(<2 x i16> %x, <2 x i16> %y, <2 x i16> <i16 8, i16 8>)
  ; CHECK-SPIRV: ReturnValue [[CALL_VEC_INT_16]]
  ret <2 x i16> %0
}

; Just check that the function for v2i16 was generated as such - we've checked the logic for another type.
; CHECK-SPIRV: Function [[TYPE_VEC_INT_16]] [[NAME_FSHR_FUNC_VEC_INT_16]] {{[0-9]+}} [[TYPE_FSHR_FUNC_VEC_INT_16]]
; CHECK-SPIRV: FunctionParameter [[TYPE_VEC_INT_16]] [[X_ARG:[0-9]+]]
; CHECK-SPIRV: FunctionParameter [[TYPE_VEC_INT_16]] [[Y_ARG:[0-9]+]]
; CHECK-SPIRV: FunctionParameter [[TYPE_VEC_INT_16]] [[ROT:[0-9]+]]

; Function Attrs: nounwind readnone speculatable willreturn
declare i32 @llvm.fshr.i32(i32, i32, i32) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare i16 @llvm.fshr.i16(i16, i16, i16) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare <2 x i16> @llvm.fshr.v2i16(<2 x i16>, <2 x i16>, <2 x i16>) #1

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
