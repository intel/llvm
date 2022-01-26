; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-SPV-IR
; RUN: llvm-spirv %t.rev.bc -o %t.rev.spv
; RUN: spirv-val %t.rev.spv

; CHECK-SPIRV-DAG: 10 TypeImage {{[0-9]*}} 6 1 0 0 0 0 0 0
; CHECK-SPIRV-DAG: 10 TypeImage {{[0-9]*}} 6 1 0 0 0 0 0 1
; CHECK-SPIRV-NOT: 10 TypeImage {{[0-9]*}} 6 1 0 0 0 0 0 0
; CHECK-SPIRV: ImageSampleExplicitLod
; CHECK-SPIRV: ImageWrite

; ModuleID = 'image.bc'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

%opencl.image2d_t = type opaque

; Function Attrs: nounwind
define spir_kernel void @image_copy(%opencl.image2d_t addrspace(1)* readnone %image1, %opencl.image2d_t addrspace(1)* %image2) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !5 !kernel_arg_type_qual !4 {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0) #3
  %conv = trunc i64 %call to i32
  %call1 = tail call spir_func i64 @_Z13get_global_idj(i32 1) #3
  %conv2 = trunc i64 %call1 to i32
  %vecinit = insertelement <2 x i32> undef, i32 %conv, i32 0
  %vecinit3 = insertelement <2 x i32> %vecinit, i32 %conv2, i32 1
  %call4 = tail call spir_func <4 x float> @_Z11read_imagef11ocl_image2d11ocl_samplerDv2_i(%opencl.image2d_t addrspace(1)* %image1, i32 20, <2 x i32> %vecinit3) #3
  tail call spir_func void @_Z12write_imagef11ocl_image2dDv2_iDv4_f(%opencl.image2d_t addrspace(1)* %image2, <2 x i32> %vecinit3, <4 x float> %call4) #4
; CHECK-LLVM: call spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iDv4_f(%opencl.image2d_wo_t addrspace(1)* %image2, <2 x i32> %vecinit3, <4 x float> %call4) #0
; CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iDv4_f(%spirv.Image._void_1_0_0_0_0_0_1 addrspace(1)* %image2, <2 x i32> %vecinit3, <4 x float> %call4) #0
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z13get_global_idj(i32) #1

; Function Attrs: nounwind readnone
declare spir_func <4 x float> @_Z11read_imagef11ocl_image2d11ocl_samplerDv2_i(%opencl.image2d_t addrspace(1)*, i32, <2 x i32>) #1

declare spir_func void @_Z12write_imagef11ocl_image2dDv2_iDv4_f(%opencl.image2d_t addrspace(1)*, <2 x i32>, <4 x float>) #2
; CHECK-LLVM: declare spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iDv4_f(%opencl.image2d_wo_t addrspace(1)*, <2 x i32>, <4 x float>)
; CHECK-SPV-IR: declare spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iDv4_f(%spirv.Image._void_1_0_0_0_0_0_1 addrspace(1)*, <2 x i32>, <4 x float>)

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!6}
!opencl.used.extensions = !{!7}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!7}
!llvm.ident = !{!9}

!1 = !{i32 1, i32 1}
!2 = !{!"read_only", !"write_only"}
!3 = !{!"image2d_t", !"image2d_t"}
!4 = !{!"", !""}
!5 = !{!"image2d_t", !"image2d_t"}
!6 = !{i32 2, i32 0}
!7 = !{}
!8 = !{!"cl_images"}
!9 = !{!"clang version 3.4 "}
