; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64"

%opencl.image2d_ro_t = type opaque
%opencl.sampler_t = type opaque

define internal spir_func <4 x float> @user_fn(%opencl.image2d_ro_t addrspace(1)* %I, %opencl.sampler_t addrspace(2)* %S, <2 x float> %Pos) {
entry:
  %call = tail call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %I, %opencl.sampler_t addrspace(2)* %S, <2 x float> noundef %Pos) #0
  ret <4 x float> %call
}

declare spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*, <2 x float>) local_unnamed_addr

define hidden spir_kernel void @kernel_fn(float addrspace(1)* %0, %opencl.image2d_ro_t addrspace(1)* %1, %opencl.sampler_t addrspace(2)* %2, i32 %3, i32 %4) local_unnamed_addr {
entry:
  %5 = call spir_func <4 x float> @user_fn(%opencl.image2d_ro_t addrspace(1)* %1, %opencl.sampler_t addrspace(2)* %2, <2 x float> undef)
  %add16 = add nsw i32 undef, undef
  ret void
}

attributes #0 = { convergent nounwind readonly willreturn }

!opencl.ocl.version = !{!0}

!0 = !{i32 2, i32 0}
