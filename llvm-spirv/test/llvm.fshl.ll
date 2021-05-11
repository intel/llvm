; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind readnone
define spir_func i32 @Test(i32 %x, i32 %y) local_unnamed_addr #0 {
entry:
  %0 = call i32 @llvm.fshl.i32(i32 %x, i32 %y, i32 8)
  ret i32 %0
}

; CHECK: TypeInt [[TYPE_INT:[0-9]+]] 32 0
; CHECK-DAG: Constant [[TYPE_INT]] [[CONST_ROTATE:[0-9]+]] 8
; CHECK-DAG: Constant [[TYPE_INT]] [[CONST_TYPE_SIZE:[0-9]+]] 32
; CHECK-DAG: Constant [[TYPE_INT]] [[CONST_0:[0-9]+]] 0
; CHECK: TypeFunction [[TYPE_ORIG_FUNC:[0-9]+]] [[TYPE_INT]] [[TYPE_INT]] [[TYPE_INT]]
; CHECK: TypeFunction [[TYPE_FSHL_FUNC:[0-9]+]] [[TYPE_INT]] [[TYPE_INT]] [[TYPE_INT]] [[TYPE_INT]]
; CHECK: TypeBool [[TYPE_BOOL:[0-9]+]]

; CHECK: Function [[TYPE_INT]] {{[0-9]+}} {{[0-9]+}} [[TYPE_ORIG_FUNC]]
; CHECK: FunctionParameter [[TYPE_INT]] [[X:[0-9]+]]
; CHECK: FunctionParameter [[TYPE_INT]] [[Y:[0-9]+]]
; CHECK: FunctionCall [[TYPE_INT]] [[CALL:[0-9]+]] [[FSHL_FUNC:[0-9]+]] [[X]] [[Y]] [[CONST_ROTATE]]
; CHECK: ReturnValue [[CALL]]

; CHECK: Function [[TYPE_INT]] [[FSHL_FUNC]] {{[0-9]+}} [[TYPE_FSHL_FUNC]]
; CHECK: FunctionParameter [[TYPE_INT]] [[X_FSHL:[0-9]+]]
; CHECK: FunctionParameter [[TYPE_INT]] [[Y_FSHL:[0-9]+]]
; CHECK: FunctionParameter [[TYPE_INT]] [[ROT:[0-9]+]]

; CHECK: Label [[MAIN_BB:[0-9]+]]
; CHECK: UMod [[TYPE_INT]] [[ROTATE_MOD_SIZE:[0-9]+]] [[ROT]] [[CONST_TYPE_SIZE]]
; CHECK: IEqual [[TYPE_BOOL]] [[ZERO_COND:[0-9]+]] [[ROTATE_MOD_SIZE]] [[CONST_0]]
; CHECK: BranchConditional [[ZERO_COND]] [[PHI_BB:[0-9]+]] [[ROTATE_BB:[0-9]+]]

; CHECK: Label [[ROTATE_BB]]
; CHECK: ShiftLeftLogical [[TYPE_INT]] [[X_SHIFT_LEFT:[0-9]+]] [[X_FSHL]] [[ROTATE_MOD_SIZE]]
; CHECK: ISub [[TYPE_INT]] [[NEG_ROTATE:[0-9]+]] [[CONST_TYPE_SIZE]] [[ROTATE_MOD_SIZE]]
; CHECK: ShiftRightLogical [[TYPE_INT]] [[Y_SHIFT_RIGHT:[0-9]+]] [[Y_FSHL]] [[NEG_ROTATE]]
; CHECK: BitwiseOr [[TYPE_INT]] [[FSHL_RESULT:[0-9]+]] [[X_SHIFT_LEFT]] [[Y_SHIFT_RIGHT]]
; CHECK: Branch [[PHI_BB]]

; CHECK: Label [[PHI_BB]]
; CHECK: Phi [[TYPE_INT]] [[PHI_INST:[0-9]+]] [[FSHL_RESULT]] [[ROTATE_BB]] [[X_FSHL]] [[MAIN_BB]]
; CHECK: ReturnValue [[PHI_INST]]

; Function Attrs: nounwind readnone speculatable willreturn
declare i32 @llvm.fshl.i32(i32, i32, i32) #1

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
