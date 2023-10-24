; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s

; This test checks following relational builtins with scalar type

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z8isfinitef(float) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z5isinff(float) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z5isnanf(float) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z8isnormalf(float) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z7signbitf(float) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func <4 x i32> @_Z8isnormalDv4_f(<4 x float> noundef) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func <4 x i32> @_Z8isfiniteDv4_f(<4 x float> noundef) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func <4 x i32> @_Z5isnanDv4_f(<4 x float> noundef) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func <4 x i32> @_Z5isinfDv4_f(<4 x float> noundef) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func <4 x i32> @_Z7signbitDv4_f(<4 x float> noundef) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define dso_local spir_kernel void @math_kernel_scalar(ptr addrspace(4) nocapture writeonly %out, float %f) local_unnamed_addr #0 {
entry:
; CHECK: [[DATA0:%.*]] = call spir_func i32 @_Z8isfinitef(float [[ARG0:%.*]])
; CHECK-NEXT: [[DATA1:%.*]] = trunc i32 [[DATA0]] to i1
; CHECK-NEXT: [[CALL0:%.*]] = select i1 [[DATA1]], i32 1, i32 0
  %call = tail call spir_func i32 @_Z8isfinitef(float %f) #2

; CHECK: [[DATA2:%.*]] = call spir_func i32 @_Z5isinff(float [[ARG0]])
; CHECK-NEXT: [[DATA3:%.*]] = trunc i32 [[DATA2]] to i1
; CHECK-NEXT: [[CALL1:%.*]] = select i1 [[DATA3]], i32 1, i32 0
  %call1 = tail call spir_func i32 @_Z5isinff(float %f) #2
  %add = add nsw i32 %call1, %call

; CHECK: [[DATA4:%.*]] = call spir_func i32 @_Z5isnanf(float [[ARG0]])
; CHECK-NEXT: [[DATA5:%.*]] = trunc i32 [[DATA4]] to i1
; CHECK-NEXT: [[CALL2:%.*]] = select i1 [[DATA5]], i32 1, i32 0
  %call2 = tail call spir_func i32 @_Z5isnanf(float %f) #2
  %add3 = add nsw i32 %add, %call2

; CHECK: [[DATA6:%.*]] = call spir_func i32 @_Z8isnormalf(float [[ARG0]])
; CHECK-NEXT: [[DATA7:%.*]] = trunc i32 [[DATA6]] to i1
; CHECK-NEXT: [[CALL3:%.*]] = select i1 [[DATA7]], i32 1, i32 0
  %call4 = tail call spir_func i32 @_Z8isnormalf(float %f) #2
  %add5 = add nsw i32 %add3, %call4

; CHECK: [[DATA8:%.*]] = call spir_func i32 @_Z7signbitf(float [[ARG0]])
; CHECK-NEXT: [[DATA9:%.*]] = trunc i32 [[DATA8]] to i1
; CHECK-NEXT: [[CALL4:%.*]] = select i1 [[DATA9]], i32 1, i32 0
  %call6 = tail call spir_func i32 @_Z7signbitf(float %f) #2
  %add7 = add nsw i32 %add5, %call6

  %arg1 = alloca <4 x float>, align 16
  %v = load <4 x float>, ptr %arg1, align 16
; CHECK: [[DATA10:%.*]] = call spir_func <4 x i32> @_Z8isnormalDv4_f(<4 x float> [[ARG1:%.*]]) #0
; CHECK-NEXT: [[DATA11:%.*]] = trunc <4 x i32> [[DATA10]] to <4 x i8>
; CHECK-NEXT: [[DATA12:%.*]] = trunc <4 x i8> [[DATA11]] to <4 x i1>
; CHECK-NEXT: [[CALL5:%.*]] = select <4 x i1> [[DATA12]], <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> zeroinitializer
  %call7 = tail call spir_func <4 x i32> @_Z8isnormalDv4_f(<4 x float> noundef %v) #2

; CHECK: [[DATA13:%.*]] = call spir_func <4 x i32> @_Z8isfiniteDv4_f(<4 x float> [[ARG1]]) #0
; CHECK-NEXT: [[DATA14:%.*]] = trunc <4 x i32> [[DATA13]] to <4 x i8>
; CHECK-NEXT: [[DATA15:%.*]] = trunc <4 x i8> [[DATA14]] to <4 x i1>
; CHECK-NEXT: [[CALL6:%.*]] = select <4 x i1> [[DATA15]], <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> zeroinitializer
  %call8 = tail call spir_func <4 x i32> @_Z8isfiniteDv4_f(<4 x float> noundef %v) #2

; CHECK: [[DATA16:%.*]] = call spir_func <4 x i32> @_Z5isnanDv4_f(<4 x float> [[ARG1]]) #0
; CHECK-NEXT: [[DATA17:%.*]] = trunc <4 x i32> [[DATA16]] to <4 x i8>
; CHECK-NEXT: [[DATA18:%.*]] = trunc <4 x i8> [[DATA17]] to <4 x i1>
; CHECK-NEXT: [[CALL7:%.*]] = select <4 x i1> [[DATA18]], <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> zeroinitializer
  %call9 = tail call spir_func <4 x i32> @_Z5isnanDv4_f(<4 x float> noundef %v) #2

; CHECK: [[DATA19:%.*]] = call spir_func <4 x i32> @_Z5isinfDv4_f(<4 x float> [[ARG1]]) #0
; CHECK-NEXT: [[DATA20:%.*]] = trunc <4 x i32> [[DATA19]] to <4 x i8>
; CHECK-NEXT: [[DATA21:%.*]] = trunc <4 x i8> [[DATA20]] to <4 x i1>
; CHECK-NEXT: [[CALL8:%.*]] = select <4 x i1> [[DATA21]], <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> zeroinitializer
  %call10 = tail call spir_func <4 x i32> @_Z5isinfDv4_f(<4 x float> noundef %v) #2

; CHECK: [[DATA22:%.*]] = call spir_func <4 x i32> @_Z7signbitDv4_f(<4 x float> [[ARG1]]) #0
; CHECK-NEXT: [[DATA23:%.*]] = trunc <4 x i32> [[DATA22]] to <4 x i8>
; CHECK-NEXT: [[DATA24:%.*]] = trunc <4 x i8> [[DATA23]] to <4 x i1>
; CHECK-NEXT: [[CALL9:%.*]] = select <4 x i1> [[DATA24]], <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> zeroinitializer
  %call11 = tail call spir_func <4 x i32> @_Z7signbitDv4_f(<4 x float> noundef %v) #2
  ret void
}

attributes #0 = { convergent mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { convergent mustprogress nofree nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind willreturn memory(none) }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 3, i32 0}
!2 = !{!"clang version 16.0.0"}
!3 = !{i32 1, i32 1}
!4 = !{!"none", !"none"}
!5 = !{!"long*", !"double*"}
!6 = !{!"", !""}
!7 = !{!8, !8, i64 0}
!8 = !{!"double", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"long", !9, i64 0}

