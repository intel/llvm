target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local spir_kernel void @OpTypeFloat64(double noundef %a, ptr addrspace(1) nocapture noundef readonly align 8 %b, ptr addrspace(1) nocapture noundef writeonly align 8 %out, ptr addrspace(3) nocapture noundef writeonly align 8 %loc) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %mul = fmul double %a, %a
  store double %mul, ptr addrspace(3) %loc, align 8, !tbaa !8
  %0 = load double, ptr addrspace(1) %b, align 8, !tbaa !8
  %1 = tail call double @llvm.fmuladd.f64(double %0, double %0, double %mul)
  store double %1, ptr addrspace(1) %out, align 8, !tbaa !8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!opencl.ocl.version = !{!2}
!opencl.spir.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{!"clang version 18.0.0git (https://github.com/intel/llvm.git 9f23932a52e62fdba649a3d0e4dc4e5cf1ed7878)"}
!4 = !{i32 0, i32 1, i32 1, i32 3}
!5 = !{!"none", !"none", !"none", !"none"}
!6 = !{!"double", !"double*", !"double*", !"double*"}
!7 = !{!"", !"", !"", !""}
!8 = !{!9, !9, i64 0}
!9 = !{!"double", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
