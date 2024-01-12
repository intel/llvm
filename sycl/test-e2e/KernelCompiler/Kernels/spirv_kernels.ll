target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

%struct.S = type { i32, float, ptr addrspace(1), %struct.Inner }
%struct.Inner = type { i32, float, ptr addrspace(1) }

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define dso_local spir_kernel void @my_kernel(ptr addrspace(1) nocapture noundef readonly align 4 %in, ptr addrspace(1) nocapture noundef writeonly align 4 %out) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #5
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %call
  %0 = load i32, ptr addrspace(1) %arrayidx, align 4, !tbaa !8
  %mul = shl nsw i32 %0, 1
  %add = add nsw i32 %mul, 100
  %arrayidx1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %call
  store i32 %add, ptr addrspace(1) %arrayidx1, align 4, !tbaa !8
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local spir_func i64 @_Z13get_global_idj(i32 noundef) local_unnamed_addr #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none)
define dso_local spir_kernel void @OpTypeStruct(ptr nocapture noundef readonly byval(%struct.S) align 8 %in, ptr addrspace(1) nocapture noundef align 8 %out) local_unnamed_addr #2 !kernel_arg_addr_space !12 !kernel_arg_access_qual !5 !kernel_arg_type !13 !kernel_arg_base_type !13 !kernel_arg_type_qual !7 {
entry:
  %0 = load i32, ptr %in, align 8, !tbaa !14
  %mul = shl nsw i32 %0, 1
  store i32 %mul, ptr addrspace(1) %out, align 8, !tbaa !14
  %f = getelementptr inbounds %struct.S, ptr %in, i64 0, i32 1
  %1 = load float, ptr %f, align 4, !tbaa !19
  %mul2 = fmul float %1, 2.000000e+00
  %f3 = getelementptr inbounds %struct.S, ptr addrspace(1) %out, i64 0, i32 1
  store float %mul2, ptr addrspace(1) %f3, align 4, !tbaa !19
  %p = getelementptr inbounds %struct.S, ptr %in, i64 0, i32 2
  %2 = load i64, ptr %p, align 8, !tbaa !20
  %3 = inttoptr i64 %2 to ptr addrspace(1)
  %4 = load i32, ptr addrspace(1) %3, align 4, !tbaa !8
  %mul4 = shl nsw i32 %4, 1
  %p5 = getelementptr inbounds %struct.S, ptr addrspace(1) %out, i64 0, i32 2
  %5 = load i64, ptr addrspace(1) %p5, align 8, !tbaa !20
  %6 = inttoptr i64 %5 to ptr addrspace(1)
  store i32 %mul4, ptr addrspace(1) %6, align 4, !tbaa !8
  %inner = getelementptr inbounds %struct.S, ptr %in, i64 0, i32 3
  %7 = load i32, ptr %inner, align 8, !tbaa !21
  %mul7 = shl nsw i32 %7, 1
  %inner8 = getelementptr inbounds %struct.S, ptr addrspace(1) %out, i64 0, i32 3
  store i32 %mul7, ptr addrspace(1) %inner8, align 8, !tbaa !21
  %f11 = getelementptr inbounds %struct.S, ptr %in, i64 0, i32 3, i32 1
  %8 = load float, ptr %f11, align 4, !tbaa !22
  %mul12 = fmul float %8, 2.000000e+00
  %f14 = getelementptr inbounds %struct.S, ptr addrspace(1) %out, i64 0, i32 3, i32 1
  store float %mul12, ptr addrspace(1) %f14, align 4, !tbaa !22
  %p16 = getelementptr inbounds %struct.S, ptr %in, i64 0, i32 3, i32 2
  %9 = load i64, ptr %p16, align 8, !tbaa !23
  %10 = inttoptr i64 %9 to ptr addrspace(1)
  %11 = load i32, ptr addrspace(1) %10, align 4, !tbaa !8
  %mul17 = shl nsw i32 %11, 1
  %p19 = getelementptr inbounds %struct.S, ptr addrspace(1) %out, i64 0, i32 3, i32 2
  %12 = load i64, ptr addrspace(1) %p19, align 8, !tbaa !23
  %13 = inttoptr i64 %12 to ptr addrspace(1)
  store i32 %mul17, ptr addrspace(1) %13, align 4, !tbaa !8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local spir_kernel void @OpTypeInt8(i8 noundef signext %a, ptr addrspace(1) nocapture noundef readonly align 1 %b, ptr addrspace(1) nocapture noundef writeonly align 1 %out, ptr addrspace(3) nocapture noundef writeonly align 1 %loc) local_unnamed_addr #3 !kernel_arg_addr_space !24 !kernel_arg_access_qual !25 !kernel_arg_type !26 !kernel_arg_base_type !26 !kernel_arg_type_qual !27 {
entry:
  %mul = mul i8 %a, %a
  store i8 %mul, ptr addrspace(3) %loc, align 1, !tbaa !28
  %0 = load i8, ptr addrspace(1) %b, align 1, !tbaa !28
  %mul6 = mul i8 %0, %0
  %add = add i8 %mul6, %mul
  store i8 %add, ptr addrspace(1) %out, align 1, !tbaa !28
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local spir_kernel void @OpTypeInt16(i16 noundef signext %a, ptr addrspace(1) nocapture noundef readonly align 2 %b, ptr addrspace(1) nocapture noundef writeonly align 2 %out, ptr addrspace(3) nocapture noundef writeonly align 2 %loc) local_unnamed_addr #3 !kernel_arg_addr_space !24 !kernel_arg_access_qual !25 !kernel_arg_type !29 !kernel_arg_base_type !29 !kernel_arg_type_qual !27 {
entry:
  %mul = mul i16 %a, %a
  store i16 %mul, ptr addrspace(3) %loc, align 2, !tbaa !30
  %0 = load i16, ptr addrspace(1) %b, align 2, !tbaa !30
  %mul6 = mul i16 %0, %0
  %add = add i16 %mul6, %mul
  store i16 %add, ptr addrspace(1) %out, align 2, !tbaa !30
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local spir_kernel void @OpTypeInt32(i32 noundef %a, ptr addrspace(1) nocapture noundef readonly align 4 %b, ptr addrspace(1) nocapture noundef writeonly align 4 %out, ptr addrspace(3) nocapture noundef writeonly align 4 %loc) local_unnamed_addr #3 !kernel_arg_addr_space !24 !kernel_arg_access_qual !25 !kernel_arg_type !32 !kernel_arg_base_type !32 !kernel_arg_type_qual !27 {
entry:
  %mul = mul nsw i32 %a, %a
  store i32 %mul, ptr addrspace(3) %loc, align 4, !tbaa !8
  %0 = load i32, ptr addrspace(1) %b, align 4, !tbaa !8
  %mul1 = mul nsw i32 %0, %0
  %add = add nuw nsw i32 %mul1, %mul
  store i32 %add, ptr addrspace(1) %out, align 4, !tbaa !8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local spir_kernel void @OpTypeInt64(i64 noundef %a, ptr addrspace(1) nocapture noundef readonly align 8 %b, ptr addrspace(1) nocapture noundef writeonly align 8 %out, ptr addrspace(3) nocapture noundef writeonly align 8 %loc) local_unnamed_addr #3 !kernel_arg_addr_space !24 !kernel_arg_access_qual !25 !kernel_arg_type !33 !kernel_arg_base_type !33 !kernel_arg_type_qual !27 {
entry:
  %mul = mul nsw i64 %a, %a
  store i64 %mul, ptr addrspace(3) %loc, align 8, !tbaa !34
  %0 = load i64, ptr addrspace(1) %b, align 8, !tbaa !34
  %mul1 = mul nsw i64 %0, %0
  %add = add nuw nsw i64 %mul1, %mul
  store i64 %add, ptr addrspace(1) %out, align 8, !tbaa !34
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local spir_kernel void @OpTypeFloat32(float noundef %a, ptr addrspace(1) nocapture noundef readonly align 4 %b, ptr addrspace(1) nocapture noundef writeonly align 4 %out, ptr addrspace(3) nocapture noundef writeonly align 4 %loc) local_unnamed_addr #3 !kernel_arg_addr_space !24 !kernel_arg_access_qual !25 !kernel_arg_type !35 !kernel_arg_base_type !35 !kernel_arg_type_qual !27 {
entry:
  %mul = fmul float %a, %a
  store float %mul, ptr addrspace(3) %loc, align 4, !tbaa !36
  %0 = load float, ptr addrspace(1) %b, align 4, !tbaa !36
  %1 = tail call float @llvm.fmuladd.f32(float %0, float %0, float %mul)
  store float %1, ptr addrspace(1) %out, align 4, !tbaa !36
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #4

attributes #0 = { convergent mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" }
attributes #1 = { convergent mustprogress nofree nounwind willreturn memory(none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!opencl.ocl.version = !{!2}
!opencl.spir.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{!"clang version 18.0.0git (https://github.com/intel/llvm.git 9f23932a52e62fdba649a3d0e4dc4e5cf1ed7878)"}
!4 = !{i32 1, i32 1}
!5 = !{!"none", !"none"}
!6 = !{!"int*", !"int*"}
!7 = !{!"", !""}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{i32 0, i32 1}
!13 = !{!"struct S", !"struct S*"}
!14 = !{!15, !9, i64 0}
!15 = !{!"S", !9, i64 0, !16, i64 4, !17, i64 8, !18, i64 16}
!16 = !{!"float", !10, i64 0}
!17 = !{!"long", !10, i64 0}
!18 = !{!"Inner", !9, i64 0, !16, i64 4, !17, i64 8}
!19 = !{!15, !16, i64 4}
!20 = !{!15, !17, i64 8}
!21 = !{!15, !9, i64 16}
!22 = !{!15, !16, i64 20}
!23 = !{!15, !17, i64 24}
!24 = !{i32 0, i32 1, i32 1, i32 3}
!25 = !{!"none", !"none", !"none", !"none"}
!26 = !{!"char", !"char*", !"char*", !"char*"}
!27 = !{!"", !"", !"", !""}
!28 = !{!10, !10, i64 0}
!29 = !{!"short", !"short*", !"short*", !"short*"}
!30 = !{!31, !31, i64 0}
!31 = !{!"short", !10, i64 0}
!32 = !{!"int", !"int*", !"int*", !"int*"}
!33 = !{!"long", !"long*", !"long*", !"long*"}
!34 = !{!17, !17, i64 0}
!35 = !{!"float", !"float*", !"float*", !"float*"}
!36 = !{!16, !16, i64 0}
