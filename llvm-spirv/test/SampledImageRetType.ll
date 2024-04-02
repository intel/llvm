; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

%opencl.image1d_ro_t = type opaque
;CHECK: TypeImage [[image1d_t:[0-9]+]]
%opencl.sampler_t = type opaque
;CHECK: TypeSampler [[sampler_t:[0-9]+]]
;CHECK: TypeSampledImage [[sampled_image_t:[0-9]+]]

declare dso_local spir_func ptr addrspace(4) @_Z20__spirv_SampledImageI14ocl_image1d_roPvET0_T_11ocl_sampler(ptr addrspace(1), ptr addrspace(2)) local_unnamed_addr #2

declare dso_local spir_func <4 x float> @_Z30__spirv_ImageSampleExplicitLodIPvDv4_fiET0_T_T1_if(ptr addrspace(4), i32, i32, float) local_unnamed_addr #2

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(2) constant <3 x i64>, align 32

define weak_odr dso_local spir_kernel void @_ZTS17image_kernel_readILi1EE(ptr addrspace(1), ptr addrspace(2)) {
;CHECK: Function
;CHECK: FunctionParameter [[image1d_t]] [[image:[0-9]+]]
;CHECK: FunctionParameter [[sampler_t]] [[sampler:[0-9]+]]
  %3 = load <3 x i64>, ptr addrspace(2) @__spirv_BuiltInGlobalInvocationId, align 32
  %4 = extractelement <3 x i64> %3, i64 0
  %5 = trunc i64 %4 to i32
  %6 = tail call spir_func ptr addrspace(4) @_Z20__spirv_SampledImageI14ocl_image1d_roPvET0_T_11ocl_sampler(ptr addrspace(1) %0, ptr addrspace(2) %1)
  %7 = tail call spir_func <4 x float> @_Z30__spirv_ImageSampleExplicitLodIPvDv4_fiET0_T_T1_if(ptr addrspace(4) %6, i32 %5, i32 2, float 0.000000e+00)

;CHECK: SampledImage [[sampled_image_t]] [[sampled_image:[0-9]+]] [[image]] [[sampler]]
;CHECK: ImageSampleExplicitLod {{[0-9]+}} {{[0-9]+}} [[sampled_image]] {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}

  ret void
}


!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 9.0.0 (f66f659153bf7052f47a80bf0bd67895c7b9a8b3)"}
