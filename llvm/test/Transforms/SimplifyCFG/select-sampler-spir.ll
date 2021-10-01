; RUN: opt < %s -simplifycfg -sink-common-insts -S | FileCheck %s
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir-unknown-unknown"

%opencl.image3d_ro_t = type opaque
%opencl.sampler_t = type opaque

; CHECK-LABEL: @K(
; CHECK-NOT: select i1 {{%.+}}, %opencl.sampler_t
; CHECK-NOT: select i1 {{%.+}}, %opencl.image

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn
define dso_local spir_kernel void @K(i32 addrspace(1)* nocapture readonly %A, %opencl.sampler_t addrspace(2)* %S, %opencl.image3d_ro_t addrspace(1)* %image, %opencl.image3d_ro_t addrspace(1)* nocapture readnone %image1, %opencl.sampler_t addrspace(2)* %S1, <4 x float> addrspace(1)* nocapture %tap) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %call = call spir_func i32 @_Z13get_global_idj(i32 0) #3
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %A, i32 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %conv = sitofp i32 %0 to float
  %splat.splatinsert = insertelement <4 x float> poison, float %conv, i32 0
  %splat.splat = shufflevector <4 x float> %splat.splatinsert, <4 x float> poison, <4 x i32> zeroinitializer
  %1 = and i32 %0, 1
  %tobool.not = icmp eq i32 %1, 0
  br i1 %tobool.not, label %cond.false, label %cond.true

cond.true:                                        ; preds = %entry
  %call2 = call spir_func <4 x float> @_Z11read_imagef14ocl_image3d_ro11ocl_samplerDv4_f(%opencl.image3d_ro_t addrspace(1)* %image, %opencl.sampler_t addrspace(2)* %S, <4 x float> %splat.splat) #4
  br label %cond.end

cond.false:                                       ; preds = %entry
  %call3 = call spir_func <4 x float> @_Z11read_imagef14ocl_image3d_ro11ocl_samplerDv4_f(%opencl.image3d_ro_t addrspace(1)* %image, %opencl.sampler_t addrspace(2)* %S1, <4 x float> %splat.splat) #4
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi <4 x float> [ %call2, %cond.true ], [ %call3, %cond.false ]
  %arrayidx4 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %tap, i32 %call
  store <4 x float> %cond, <4 x float> addrspace(1)* %arrayidx4, align 16
  ret void
}

; Function Attrs: convergent nounwind readnone willreturn
declare spir_func i32 @_Z13get_global_idj(i32) #1

; Function Attrs: convergent nounwind readonly willreturn
declare spir_func <4 x float> @_Z11read_imagef14ocl_image3d_ro11ocl_samplerDv4_f(%opencl.image3d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*, <4 x float>) #2

attributes #0 = { convergent norecurse nounwind "frame-pointer"="none" "min-legal-vector-width"="128" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { convergent nounwind readnone willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind readonly willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent nounwind readnone willreturn }
attributes #4 = { convergent nounwind readonly willreturn }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 14.0.0 (https://github.com/intel/llvm.git 183caf354e603f8935163874374df716a25cf33d)"}
!3 = !{i32 1, i32 0, i32 1, i32 1, i32 0, i32 1}
!4 = !{!"none", !"none", !"read_only", !"read_only", !"none", !"none"}
!5 = !{!"int*", !"sampler_t", !"image3d_t", !"image3d_t", !"sampler_t", !"float4*"}
!6 = !{!"int*", !"sampler_t", !"image3d_t", !"image3d_t", !"sampler_t", !"float __attribute__((ext_vector_type(4)))*"}
!7 = !{!"", !"", !"", !"", !"", !""}
