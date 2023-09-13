;; Test SPIR-V opaque types
;;
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spv.txt
; RUN: FileCheck < %t.spv.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.from-llvm.spv
; RUN: llvm-spirv -to-binary %t.spv.txt -o %t.from-text.spv
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv --spirv-target-env=SPV-IR -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM-SPIRV

; Check that produced SPIR-V friendly IR is correctly recognized
; RUN: llvm-spirv %t.rev.bc -spirv-text -o %t.spv.txt
; RUN: FileCheck < %t.spv.txt %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: 2 Capability Float16
; CHECK-SPIRV: 2 Capability ImageBasic
; CHECK-SPIRV: 2 Capability ImageReadWrite
; CHECK-SPIRV: 2 Capability Pipes
; CHECK-SPIRV: 2 Capability DeviceEnqueue

; CHECK-SPIRV-DAG: 2 TypeVoid [[VOID:[0-9]+]]
; CHECK-SPIRV-DAG: 4 TypeInt [[INT:[0-9]+]] 32 0
; CHECK-SPIRV-DAG: 3 TypeFloat [[HALF:[0-9]+]] 16
; CHECK-SPIRV-DAG: 3 TypeFloat [[FLOAT:[0-9]+]] 32
; CHECK-SPIRV-DAG: 3 TypePipe [[PIPE_RD:[0-9]+]] 0
; CHECK-SPIRV-DAG: 3 TypePipe [[PIPE_WR:[0-9]+]] 1
; CHECK-SPIRV-DAG: 10 TypeImage [[IMG1D_RD:[0-9]+]] [[VOID]] 0 0 0 0 0 0 0
; CHECK-SPIRV-DAG: 10 TypeImage [[IMG2D_RD:[0-9]+]] [[INT]] 1 0 0 0 0 0 0
; CHECK-SPIRV-DAG: 10 TypeImage [[IMG3D_RD:[0-9]+]] [[INT]] 2 0 0 0 0 0 0
; CHECK-SPIRV-DAG: 10 TypeImage [[IMG2DD_RD:[0-9]+]] [[FLOAT]] 1 1 0 0 0 0 0
; CHECK-SPIRV-DAG: 10 TypeImage [[IMG2DA_RD:[0-9]+]] [[HALF]] 1 0 1 0 0 0 0
; CHECK-SPIRV-DAG: 10 TypeImage [[IMG1DB_RD:[0-9]+]] [[FLOAT]] 5 0 0 0 0 0 0
; CHECK-SPIRV-DAG: 10 TypeImage [[IMG1D_WR:[0-9]+]] [[VOID]] 0 0 0 0 0 0 1
; CHECK-SPIRV-DAG: 10 TypeImage [[IMG2D_RW:[0-9]+]] [[VOID]] 1 0 0 0 0 0 2
; CHECK-SPIRV-DAG: 2 TypeDeviceEvent [[DEVEVENT:[0-9]+]]
; CHECK-SPIRV-DAG: 2 TypeEvent [[EVENT:[0-9]+]]
; CHECK-SPIRV-DAG: 2 TypeQueue [[QUEUE:[0-9]+]]
; CHECK-SPIRV-DAG: 2 TypeReserveId [[RESID:[0-9]+]]
; CHECK-SPIRV-DAG: 2 TypeSampler [[SAMP:[0-9]+]]
; CHECK-SPIRV-DAG: 3 TypeSampledImage [[SAMPIMG:[0-9]+]] [[IMG2DD_RD]]

; ModuleID = 'cl-types.cl'
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-SPIRV: {{[0-9]+}} Function
; CHECK-SPIRV: 3 FunctionParameter [[PIPE_RD]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[PIPE_WR]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[IMG1D_RD]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[IMG2D_RD]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[IMG3D_RD]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[IMG2DA_RD]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[IMG1DB_RD]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[IMG1D_WR]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[IMG2D_RW]] {{[0-9]+}}

; CHECK-LLVM:        define spir_kernel void @foo(
; CHECK-LLVM-SAME:     ptr addrspace(1) %a,
; CHECK-LLVM-SAME:     ptr addrspace(1) %b,
; CHECK-LLVM-SAME:     ptr addrspace(1) %c1,
; CHECK-LLVM-SAME:     ptr addrspace(1) %d1,
; CHECK-LLVM-SAME:     ptr addrspace(1) %e1,
; CHECK-LLVM-SAME:     ptr addrspace(1) %f1,
; CHECK-LLVM-SAME:     ptr addrspace(1) %g1,
; CHECK-LLVM-SAME:     ptr addrspace(1) %c2,
; CHECK-LLVM-SAME:     ptr addrspace(1) %d3)
; CHECK-LLVM-SAME:     !kernel_arg_addr_space [[AS:![0-9]+]]
; CHECK-LLVM-SAME:     !kernel_arg_access_qual [[AQ:![0-9]+]]
; CHECK-LLVM-SAME:     !kernel_arg_type [[TYPE:![0-9]+]]
; CHECK-LLVM-SAME:     !kernel_arg_type_qual [[TQ:![0-9]+]]
; CHECK-LLVM-SAME:     !kernel_arg_base_type [[TYPE]]

; Function Attrs: nounwind readnone
define spir_kernel void @foo(
  target("spirv.Pipe", 0) %a,
  target("spirv.Pipe", 1) %b,
  target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %c1,
  target("spirv.Image", i32, 1, 0, 0, 0, 0, 0, 0) %d1,
  target("spirv.Image", i32, 2, 0, 0, 0, 0, 0, 0) %e1,
  target("spirv.Image", half, 1, 0, 1, 0, 0, 0, 0) %f1,
  target("spirv.Image", float, 5, 0, 0, 0, 0, 0, 0) %g1,
  target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 1) %c2,
  target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 2) %d3) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  ret void
}

; CHECK-SPIRV: {{[0-9]+}} Function
; CHECK-SPIRV: 3 FunctionParameter [[DEVEVENT]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[EVENT]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[QUEUE]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[RESID]] {{[0-9]+}}

; CHECK-LLVM: define spir_func void @bar(
; CHECK-LLVM:  ptr %a,
; CHECK-LLVM:  ptr %b,
; CHECK-LLVM:  ptr %c,
; CHECK-LLVM:  ptr %d)

define spir_func void @bar(
  target("spirv.DeviceEvent") %a,
  target("spirv.Event") %b,
  target("spirv.Queue") %c,
  target("spirv.ReserveId") %d) {
  ret void
}

; CHECK-SPIRV: {{[0-9]+}} Function
; CHECK-SPIRV: 3 FunctionParameter [[IMG2DD_RD]] [[IMG_ARG:[0-9]+]]
; CHECK-SPIRV: 3 FunctionParameter [[SAMP]] [[SAMP_ARG:[0-9]+]]
; CHECK-SPIRV: 5 SampledImage [[SAMPIMG]] [[SAMPIMG_VAR:[0-9]+]] [[IMG_ARG]] [[SAMP_ARG]]
; CHECK-SPIRV: 7 ImageSampleExplicitLod {{[0-9]+}} {{[0-9]+}} [[SAMPIMG_VAR]]

; CHECK-LLVM: define spir_func void @test_sampler(
; CHECK-LLVM:  ptr addrspace(1) %srcimg.coerce,
; CHECK-LLVM:  ptr addrspace(2) %s.coerce)
; CHECK-LLVM:  call spir_func float @_Z11read_imagef20ocl_image2d_depth_ro11ocl_samplerDv4_if(ptr addrspace(1) %srcimg.coerce, ptr addrspace(2) %s.coerce, <4 x i32> zeroinitializer, float 1.000000e+00)

; CHECK-LLVM-SPIRV: call spir_func target("spirv.SampledImage", float, 1, 1, 0, 0, 0, 0, 0) @_Z20__spirv_SampledImagePU3AS134__spirv_Image__float_1_1_0_0_0_0_0PU3AS215__spirv_Sampler(target("spirv.Image", float, 1, 1, 0, 0, 0, 0, 0) %srcimg.coerce, target("spirv.Sampler") %s.coerce)
; CHECK-LLVM-SPIRV: call spir_func <4 x float> @_Z38__spirv_ImageSampleExplicitLod_Rfloat4PU3AS141__spirv_SampledImage__float_1_1_0_0_0_0_0Dv4_iif(target("spirv.SampledImage", float, 1, 1, 0, 0, 0, 0, 0) %1, <4 x i32> zeroinitializer, i32 2, float 1.000000e+00)

define spir_func void @test_sampler(target("spirv.Image", float, 1, 1, 0, 0, 0, 0, 0) %srcimg.coerce,
                                    target("spirv.Sampler") %s.coerce) {
  %1 = tail call spir_func target("spirv.SampledImage", float, 1, 1, 0, 0, 0, 0, 0) @_Z20__spirv_SampledImagePU3AS1K34__spirv_Image__float_1_1_0_0_0_0_0PU3AS1K15__spirv_Sampler(target("spirv.Image", float, 1, 1, 0, 0, 0, 0, 0) %srcimg.coerce, target("spirv.Sampler") %s.coerce) #1
  %2 = tail call spir_func <4 x float> @_Z38__spirv_ImageSampleExplicitLod_Rfloat4PU3AS120__spirv_SampledImageDv4_iif(target("spirv.SampledImage", float, 1, 1, 0, 0, 0, 0, 0) %1, <4 x i32> zeroinitializer, i32 2, float 1.000000e+00) #1
  ret void
}

declare spir_func target("spirv.SampledImage", float, 1, 1, 0, 0, 0, 0, 0) @_Z20__spirv_SampledImagePU3AS1K34__spirv_Image__float_1_1_0_0_0_0_0PU3AS1K15__spirv_Sampler(target("spirv.Image", float, 1, 1, 0, 0, 0, 0, 0), target("spirv.Sampler"))

declare spir_func <4 x float> @_Z38__spirv_ImageSampleExplicitLod_Rfloat4PU3AS120__spirv_SampledImageDv4_iif(target("spirv.SampledImage", float, 1, 1, 0, 0, 0, 0, 0), <4 x i32>, i32, float)

attributes #0 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!9}
!opencl.compiler.options = !{!8}

; CHECK-LLVM-DAG: [[AS]] = !{i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1}
; CHECK-LLVM-DAG: [[AQ]] = !{!"read_only", !"write_only", !"read_only", !"read_only", !"read_only", !"read_only", !"read_only", !"write_only", !"read_write"}
; CHECK-LLVM-DAG: [[TYPE]] = !{!"pipe", !"pipe", !"image1d_t", !"image2d_t", !"image3d_t", !"image2d_array_t", !"image1d_buffer_t", !"image1d_t", !"image2d_t"}
; CHECK-LLVM-DAG: [[TQ]] = !{!"pipe", !"pipe", !"", !"", !"", !"", !"", !"", !""}

!1 = !{i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1}
!2 = !{!"read_only", !"write_only", !"read_only", !"read_only", !"read_only", !"read_only", !"read_only", !"write_only", !"read_write"}
!3 = !{!"int", !"int", !"image1d_t", !"image2d_t", !"image3d_t", !"image2d_array_t", !"image1d_buffer_t", !"image1d_t", !"image2d_t"}
!4 = !{!"int", !"int", !"image1d_t", !"image2d_t", !"image3d_t", !"image2d_array_t", !"image1d_buffer_t", !"image1d_t", !"image2d_t"}
!5 = !{!"pipe", !"pipe", !"", !"", !"", !"", !"", !"", !""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{!"cl_khr_fp16"}
!9 = !{!"cl_images"}
