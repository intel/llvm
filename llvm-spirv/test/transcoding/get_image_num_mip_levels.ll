; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc --spirv-target-env=SPV-IR
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-SPV-IR
; RUN: llvm-spirv -spirv-text %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; Generated from the following OpenCL C code:
; #pragma OPENCL EXTENSION cl_khr_mipmap_image : enable
; void test(image1d_t img1, 
;           image2d_t img2,
;           image3d_t img3,
;           image1d_array_t img4,
;           image2d_array_t img5,
;           image2d_depth_t img6,
;           image2d_array_depth_t img7)
; {
;     get_image_num_mip_levels(img1);
;     get_image_num_mip_levels(img2);
;     get_image_num_mip_levels(img3);
;     get_image_num_mip_levels(img4);
;     get_image_num_mip_levels(img5);
;     get_image_num_mip_levels(img6);
;     get_image_num_mip_levels(img7);
; }

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.image1d_ro_t = type opaque
%opencl.image2d_ro_t = type opaque
%opencl.image3d_ro_t = type opaque
%opencl.image1d_array_ro_t = type opaque
%opencl.image2d_array_ro_t = type opaque
%opencl.image2d_depth_ro_t = type opaque
%opencl.image2d_array_depth_ro_t = type opaque

; CHECK-SPIRV: TypeInt [[INT:[0-9]+]] 32
; CHECK-SPIRV: TypeImage [[IMAGE1D_T:[0-9]+]] 2 0 0 0 0 0 0 0
; CHECK-SPIRV: TypeImage [[IMAGE2D_T:[0-9]+]] 2 1 0 0 0 0 0 0
; CHECK-SPIRV: TypeImage [[IMAGE3D_T:[0-9]+]] 2 2 0 0 0 0 0 0
; CHECK-SPIRV: TypeImage [[IMAGE1D_ARRAY_T:[0-9]+]] 2 0 0 1 0 0 0 0
; CHECK-SPIRV: TypeImage [[IMAGE2D_ARRAY_T:[0-9]+]] 2 1 0 1 0 0 0 0
; CHECK-SPIRV: TypeImage [[IMAGE2D_DEPTH_T:[0-9]+]] 2 1 1 0 0 0 0 0
; CHECK-SPIRV: TypeImage [[IMAGE2D_ARRAY_DEPTH_T:[0-9]+]] 2 1 1 1 0 0 0 0

; CHECK-LLVM: %opencl.image1d_ro_t = type opaque
; CHECK-LLVM: %opencl.image2d_ro_t = type opaque
; CHECK-LLVM: %opencl.image3d_ro_t = type opaque
; CHECK-LLVM: %opencl.image1d_array_ro_t = type opaque
; CHECK-LLVM: %opencl.image2d_array_ro_t = type opaque
; CHECK-LLVM: %opencl.image2d_depth_ro_t = type opaque
; CHECK-LLVM: %opencl.image2d_array_depth_ro_t = type opaque

; CHECK-SPV-IR: %spirv.Image._void_0_0_0_0_0_0_0 = type opaque
; CHECK-SPV-IR: %spirv.Image._void_1_0_0_0_0_0_0 = type opaque
; CHECK-SPV-IR: %spirv.Image._void_2_0_0_0_0_0_0 = type opaque
; CHECK-SPV-IR: %spirv.Image._void_0_0_1_0_0_0_0 = type opaque
; CHECK-SPV-IR: %spirv.Image._void_1_0_1_0_0_0_0 = type opaque
; CHECK-SPV-IR: %spirv.Image._void_1_1_0_0_0_0_0 = type opaque
; CHECK-SPV-IR: %spirv.Image._void_1_1_1_0_0_0_0 = type opaque

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_func void @testimage1d(%opencl.image1d_ro_t addrspace(1)* %img1, %opencl.image2d_ro_t addrspace(1)* %img2, %opencl.image3d_ro_t addrspace(1)* %img3, %opencl.image1d_array_ro_t addrspace(1)* %img4, %opencl.image2d_array_ro_t addrspace(1)* %img5, %opencl.image2d_depth_ro_t addrspace(1)* %img6, %opencl.image2d_array_depth_ro_t addrspace(1)* %img7) #0 {
entry:
  %img1.addr = alloca %opencl.image1d_ro_t addrspace(1)*, align 4
  %img2.addr = alloca %opencl.image2d_ro_t addrspace(1)*, align 4
  %img3.addr = alloca %opencl.image3d_ro_t addrspace(1)*, align 4
  %img4.addr = alloca %opencl.image1d_array_ro_t addrspace(1)*, align 4
  %img5.addr = alloca %opencl.image2d_array_ro_t addrspace(1)*, align 4
  %img6.addr = alloca %opencl.image2d_depth_ro_t addrspace(1)*, align 4
  %img7.addr = alloca %opencl.image2d_array_depth_ro_t addrspace(1)*, align 4
  store %opencl.image1d_ro_t addrspace(1)* %img1, %opencl.image1d_ro_t addrspace(1)** %img1.addr, align 4
  store %opencl.image2d_ro_t addrspace(1)* %img2, %opencl.image2d_ro_t addrspace(1)** %img2.addr, align 4
  store %opencl.image3d_ro_t addrspace(1)* %img3, %opencl.image3d_ro_t addrspace(1)** %img3.addr, align 4
  store %opencl.image1d_array_ro_t addrspace(1)* %img4, %opencl.image1d_array_ro_t addrspace(1)** %img4.addr, align 4
  store %opencl.image2d_array_ro_t addrspace(1)* %img5, %opencl.image2d_array_ro_t addrspace(1)** %img5.addr, align 4
  store %opencl.image2d_depth_ro_t addrspace(1)* %img6, %opencl.image2d_depth_ro_t addrspace(1)** %img6.addr, align 4
  store %opencl.image2d_array_depth_ro_t addrspace(1)* %img7, %opencl.image2d_array_depth_ro_t addrspace(1)** %img7.addr, align 4
  %0 = load %opencl.image1d_ro_t addrspace(1)*, %opencl.image1d_ro_t addrspace(1)** %img1.addr, align 4
  %call = call spir_func i32 @_Z24get_image_num_mip_levels14ocl_image1d_ro(%opencl.image1d_ro_t addrspace(1)* %0) #2
  %1 = load %opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)** %img2.addr, align 4
  %call1 = call spir_func i32 @_Z24get_image_num_mip_levels14ocl_image2d_ro(%opencl.image2d_ro_t addrspace(1)* %1) #2
  %2 = load %opencl.image3d_ro_t addrspace(1)*, %opencl.image3d_ro_t addrspace(1)** %img3.addr, align 4
  %call2 = call spir_func i32 @_Z24get_image_num_mip_levels14ocl_image3d_ro(%opencl.image3d_ro_t addrspace(1)* %2) #2
  %3 = load %opencl.image1d_array_ro_t addrspace(1)*, %opencl.image1d_array_ro_t addrspace(1)** %img4.addr, align 4
  %call3 = call spir_func i32 @_Z24get_image_num_mip_levels20ocl_image1d_array_ro(%opencl.image1d_array_ro_t addrspace(1)* %3) #2
  %4 = load %opencl.image2d_array_ro_t addrspace(1)*, %opencl.image2d_array_ro_t addrspace(1)** %img5.addr, align 4
  %call4 = call spir_func i32 @_Z24get_image_num_mip_levels20ocl_image2d_array_ro(%opencl.image2d_array_ro_t addrspace(1)* %4) #2
  %5 = load %opencl.image2d_depth_ro_t addrspace(1)*, %opencl.image2d_depth_ro_t addrspace(1)** %img6.addr, align 4
  %call5 = call spir_func i32 @_Z24get_image_num_mip_levels20ocl_image2d_depth_ro(%opencl.image2d_depth_ro_t addrspace(1)* %5) #2
  %6 = load %opencl.image2d_array_depth_ro_t addrspace(1)*, %opencl.image2d_array_depth_ro_t addrspace(1)** %img7.addr, align 4
  %call6 = call spir_func i32 @_Z24get_image_num_mip_levels26ocl_image2d_array_depth_ro(%opencl.image2d_array_depth_ro_t addrspace(1)* %6) #2
  ret void
}

; CHECK-SPIRV: Load [[IMAGE1D_T]] [[IMAGE1D:[0-9]+]] 
; CHECK-SPIRV: ImageQueryLevels [[INT]] {{[0-9]+}} [[IMAGE1D]]
; CHECK-SPIRV: Load [[IMAGE2D_T]] [[IMAGE2D:[0-9]+]] 
; CHECK-SPIRV: ImageQueryLevels [[INT]] {{[0-9]+}} [[IMAGE2D]]
; CHECK-SPIRV: Load [[IMAGE3D_T]] [[IMAGE3D:[0-9]+]] 
; CHECK-SPIRV: ImageQueryLevels [[INT]] {{[0-9]+}} [[IMAGE3D]]
; CHECK-SPIRV: Load [[IMAGE1D_ARRAY_T]] [[IMAGE1D_ARRAY:[0-9]+]] 
; CHECK-SPIRV: ImageQueryLevels [[INT]] {{[0-9]+}} [[IMAGE1D_ARRAY]]
; CHECK-SPIRV: Load [[IMAGE2D_ARRAY_T]] [[IMAGE2D_ARRAY:[0-9]+]] 
; CHECK-SPIRV: ImageQueryLevels [[INT]] {{[0-9]+}} [[IMAGE2D_ARRAY]]
; CHECK-SPIRV: Load [[IMAGE2D_DEPTH_T]] [[IMAGE2D_DEPTH:[0-9]+]] 
; CHECK-SPIRV: ImageQueryLevels [[INT]] {{[0-9]+}} [[IMAGE2D_DEPTH]]
; CHECK-SPIRV: Load [[IMAGE2D_ARRAY_DEPTH_T]] [[IMAGE2D_ARRAY_DEPTH:[0-9]+]] 
; CHECK-SPIRV: ImageQueryLevels [[INT]] {{[0-9]+}} [[IMAGE2D_ARRAY_DEPTH]]

; CHECK-LLVM: call spir_func i32 @_Z24get_image_num_mip_levels14ocl_image1d_ro(%opencl.image1d_ro_t addrspace(1)*
; CHECK-LLVM: call spir_func i32 @_Z24get_image_num_mip_levels14ocl_image2d_ro(%opencl.image2d_ro_t addrspace(1)*
; CHECK-LLVM: call spir_func i32 @_Z24get_image_num_mip_levels14ocl_image3d_ro(%opencl.image3d_ro_t addrspace(1)*
; CHECK-LLVM: call spir_func i32 @_Z24get_image_num_mip_levels20ocl_image1d_array_ro(%opencl.image1d_array_ro_t addrspace(1)*
; CHECK-LLVM: call spir_func i32 @_Z24get_image_num_mip_levels20ocl_image2d_array_ro(%opencl.image2d_array_ro_t addrspace(1)*
; CHECK-LLVM: call spir_func i32 @_Z24get_image_num_mip_levels20ocl_image2d_depth_ro(%opencl.image2d_depth_ro_t addrspace(1)*
; CHECK-LLVM: call spir_func i32 @_Z24get_image_num_mip_levels26ocl_image2d_array_depth_ro(%opencl.image2d_array_depth_ro_t addrspace(1)*

; CHECK-SPV-IR: call spir_func i32 @_Z24__spirv_ImageQueryLevelsPU3AS133__spirv_Image__void_0_0_0_0_0_0_0(%spirv.Image._void_0_0_0_0_0_0_0 addrspace(1)*
; CHECK-SPV-IR: call spir_func i32 @_Z24__spirv_ImageQueryLevelsPU3AS133__spirv_Image__void_1_0_0_0_0_0_0(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)*
; CHECK-SPV-IR: call spir_func i32 @_Z24__spirv_ImageQueryLevelsPU3AS133__spirv_Image__void_2_0_0_0_0_0_0(%spirv.Image._void_2_0_0_0_0_0_0 addrspace(1)*
; CHECK-SPV-IR: call spir_func i32 @_Z24__spirv_ImageQueryLevelsPU3AS133__spirv_Image__void_0_0_1_0_0_0_0(%spirv.Image._void_0_0_1_0_0_0_0 addrspace(1)*
; CHECK-SPV-IR: call spir_func i32 @_Z24__spirv_ImageQueryLevelsPU3AS133__spirv_Image__void_1_0_1_0_0_0_0(%spirv.Image._void_1_0_1_0_0_0_0 addrspace(1)*
; CHECK-SPV-IR: call spir_func i32 @_Z24__spirv_ImageQueryLevelsPU3AS133__spirv_Image__void_1_1_0_0_0_0_0(%spirv.Image._void_1_1_0_0_0_0_0 addrspace(1)*
; CHECK-SPV-IR: call spir_func i32 @_Z24__spirv_ImageQueryLevelsPU3AS133__spirv_Image__void_1_1_1_0_0_0_0(%spirv.Image._void_1_1_1_0_0_0_0 addrspace(1)*

; Function Attrs: convergent
declare spir_func i32 @_Z24get_image_num_mip_levels14ocl_image1d_ro(%opencl.image1d_ro_t addrspace(1)*) #1

; Function Attrs: convergent
declare spir_func i32 @_Z24get_image_num_mip_levels14ocl_image2d_ro(%opencl.image2d_ro_t addrspace(1)*) #1

; Function Attrs: convergent
declare spir_func i32 @_Z24get_image_num_mip_levels14ocl_image3d_ro(%opencl.image3d_ro_t addrspace(1)*) #1

; Function Attrs: convergent
declare spir_func i32 @_Z24get_image_num_mip_levels20ocl_image1d_array_ro(%opencl.image1d_array_ro_t addrspace(1)*) #1

; Function Attrs: convergent
declare spir_func i32 @_Z24get_image_num_mip_levels20ocl_image2d_array_ro(%opencl.image2d_array_ro_t addrspace(1)*) #1

; Function Attrs: convergent
declare spir_func i32 @_Z24get_image_num_mip_levels20ocl_image2d_depth_ro(%opencl.image2d_depth_ro_t addrspace(1)*) #1

; Function Attrs: convergent
declare spir_func i32 @_Z24get_image_num_mip_levels26ocl_image2d_array_depth_ro(%opencl.image2d_array_depth_ro_t addrspace(1)*) #1

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
