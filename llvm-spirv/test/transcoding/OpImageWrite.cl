// RUN: %clang_cc1 -O1 -triple spir-unknown-unknown -cl-std=CL2.0 %s -finclude-default-header -emit-llvm-bc -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
// RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc --spirv-target-env=SPV-IR
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-SPV-IR

// CHECK-SPIRV: TypeVoid [[VOID_TY:[0-9]+]]
// CHECK-SPIRV: TypeImage [[IMG2D_WO_TY:[0-9]+]] [[VOID_TY]] 1 0 0 0 0 0 1
// CHECK-SPIRV: TypeImage [[IMG2D_RW_TY:[0-9]+]] [[VOID_TY]] 1 0 0 0 0 0 2
// CHECK-SPIRV: TypeImage [[IMG2D_ARRAY_WO_TY:[0-9]+]] [[VOID_TY]] 1 0 1 0 0 0 1
// CHECK-SPIRV: TypeImage [[IMG2D_ARRAY_RW_TY:[0-9]+]] [[VOID_TY]] 1 0 1 0 0 0 2
// CHECK-SPIRV: TypeImage [[IMG1D_WO_TY:[0-9]+]] [[VOID_TY]] 0 0 0 0 0 0 1
// CHECK-SPIRV: TypeImage [[IMG1D_RW_TY:[0-9]+]] [[VOID_TY]] 0 0 0 0 0 0 2
// CHECK-SPIRV: TypeImage [[IMG1D_BUFFER_WO_TY:[0-9]+]] [[VOID_TY]] 5 0 0 0 0 0 1
// CHECK-SPIRV: TypeImage [[IMG1D_BUFFER_RW_TY:[0-9]+]] [[VOID_TY]] 5 0 0 0 0 0 2
// CHECK-SPIRV: TypeImage [[IMG1D_ARRAY_WO_TY:[0-9]+]] [[VOID_TY]] 0 0 1 0 0 0 1
// CHECK-SPIRV: TypeImage [[IMG1D_ARRAY_RW_TY:[0-9]+]] [[VOID_TY]] 0 0 1 0 0 0 2
// CHECK-SPIRV: TypeImage [[IMG2D_DEPTH_WO_TY:[0-9]+]] [[VOID_TY]] 1 1 0 0 0 0 1
// CHECK-SPIRV: TypeImage [[IMG2D_ARRAY_DEPTH_WO_TY:[0-9]+]] [[VOID_TY]] 1 1 1 0 0 0 1
// CHECK-SPIRV: TypeImage [[IMG3D_WO_TY:[0-9]+]] [[VOID_TY]] 2 0 0 0 0 0 1
// CHECK-SPIRV: TypeImage [[IMG3D_RW_TY:[0-9]+]] [[VOID_TY]] 2 0 0 0 0 0 2

kernel void test_img2d(write_only image2d_t image_wo, read_write image2d_t image_rw)
{
    write_imagef(image_wo, (int2)(0,0), (float4)(0,0,0,0));
    write_imagei(image_wo, (int2)(0,0), (int4)(0,0,0,0));
    write_imagef(image_rw, (int2)(0,0), (float4)(0,0,0,0));
    write_imagei(image_rw, (int2)(0,0), (int4)(0,0,0,0));
    
    // LOD
    write_imagef(image_wo, (int2)(0,0), 0, (float4)(0,0,0,0));
    write_imagei(image_wo, (int2)(0,0), 0, (int4)(0,0,0,0));
}

// CHECK-SPIRV: FunctionParameter [[IMG2D_WO_TY]] [[IMG2D_WO:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[IMG2D_RW_TY]] [[IMG2D_RW:[0-9]+]]

// CHECK-SPIRV: ImageWrite [[IMG2D_WO]]
// CHECK-SPIRV: ImageWrite [[IMG2D_WO]]
// CHECK-SPIRV: ImageWrite [[IMG2D_RW]]
// CHECK-SPIRV: ImageWrite [[IMG2D_RW]]
// CHECK-SPIRV: ImageWrite [[IMG2D_WO]]
// CHECK-SPIRV: ImageWrite [[IMG2D_WO]]

// CHECK-LLVM: call spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei14ocl_image2d_woDv2_iDv4_i
// CHECK-LLVM: call spir_func void @_Z12write_imagef14ocl_image2d_rwDv2_iDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei14ocl_image2d_rwDv2_iDv4_i
// CHECK-LLVM: call spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iiDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei14ocl_image2d_woDv2_iiDv4_i

// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iDv4_f
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iDv4_i
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_0_0_0_0_0_2Dv2_iDv4_f
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_0_0_0_0_0_2Dv2_iDv4_i
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iDv4_fii
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iDv4_iii

kernel void test_img2d_array(write_only image2d_array_t image_wo, read_write image2d_array_t image_rw)
{
    write_imagef(image_wo, (int4)(0,0,0,0), (float4)(0,0,0,0));
    write_imagei(image_wo, (int4)(0,0,0,0), (int4)(0,0,0,0));
    write_imagef(image_rw, (int4)(0,0,0,0), (float4)(0,0,0,0));
    write_imagei(image_rw, (int4)(0,0,0,0), (int4)(0,0,0,0));
    
    // LOD
    write_imagef(image_wo, (int4)(0,0,0,0), 0, (float4)(0,0,0,0));
    write_imagei(image_wo, (int4)(0,0,0,0), 0, (int4)(0,0,0,0));
}

// CHECK-SPIRV: FunctionParameter [[IMG2D_ARRAY_WO_TY]] [[IMG2D_ARRAY_WO:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[IMG2D_ARRAY_RW_TY]] [[IMG2D_ARRAY_RW:[0-9]+]]

// CHECK-SPIRV: ImageWrite [[IMG2D_ARRAY_WO]]
// CHECK-SPIRV: ImageWrite [[IMG2D_ARRAY_WO]]
// CHECK-SPIRV: ImageWrite [[IMG2D_ARRAY_RW]]
// CHECK-SPIRV: ImageWrite [[IMG2D_ARRAY_RW]]
// CHECK-SPIRV: ImageWrite [[IMG2D_ARRAY_WO]]
// CHECK-SPIRV: ImageWrite [[IMG2D_ARRAY_WO]]

// CHECK-LLVM: call spir_func void @_Z12write_imagef20ocl_image2d_array_woDv4_iDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei20ocl_image2d_array_woDv4_iS0_
// CHECK-LLVM: call spir_func void @_Z12write_imagef20ocl_image2d_array_rwDv4_iDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei20ocl_image2d_array_rwDv4_iS0_
// CHECK-LLVM: call spir_func void @_Z12write_imagef20ocl_image2d_array_woDv4_iiDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei20ocl_image2d_array_woDv4_iiS0_

// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_0_1_0_0_0_1Dv4_iDv4_f
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_0_1_0_0_0_1Dv4_iS2_
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_0_1_0_0_0_2Dv4_iDv4_f
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_0_1_0_0_0_2Dv4_iS2_
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_0_1_0_0_0_1Dv4_iDv4_fii
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_0_1_0_0_0_1Dv4_iS2_ii

kernel void test_img1d(write_only image1d_t image_wo, read_write image1d_t image_rw)
{
    write_imagef(image_wo, 0, (float4)(0,0,0,0));
    write_imagei(image_wo, 0, (int4)(0,0,0,0));
    write_imagef(image_rw, 0, (float4)(0,0,0,0));
    write_imagei(image_rw, 0, (int4)(0,0,0,0));
    
    // LOD
    write_imagef(image_wo, 0, 0, (float4)(0,0,0,0));
    write_imagei(image_wo, 0, 0, (int4)(0,0,0,0));
}

// CHECK-SPIRV: FunctionParameter [[IMG1D_WO_TY]] [[IMG1D_WO:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[IMG1D_RW_TY]] [[IMG1D_RW:[0-9]+]]

// CHECK-SPIRV: ImageWrite [[IMG1D_WO]]
// CHECK-SPIRV: ImageWrite [[IMG1D_WO]]
// CHECK-SPIRV: ImageWrite [[IMG1D_RW]]
// CHECK-SPIRV: ImageWrite [[IMG1D_RW]]
// CHECK-SPIRV: ImageWrite [[IMG1D_WO]]
// CHECK-SPIRV: ImageWrite [[IMG1D_WO]]

// CHECK-LLVM: call spir_func void @_Z12write_imagef14ocl_image1d_woiDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei14ocl_image1d_woiDv4_i
// CHECK-LLVM: call spir_func void @_Z12write_imagef14ocl_image1d_rwiDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei14ocl_image1d_rwiDv4_i
// CHECK-LLVM: call spir_func void @_Z12write_imagef14ocl_image1d_woiiDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei14ocl_image1d_woiiDv4_i

// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_0_0_0_0_0_0_1iDv4_f
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_0_0_0_0_0_0_1iDv4_i
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_0_0_0_0_0_0_2iDv4_f
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_0_0_0_0_0_0_2iDv4_i
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_0_0_0_0_0_0_1iDv4_fii
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_0_0_0_0_0_0_1iDv4_iii

kernel void test_img1d_buffer(write_only image1d_buffer_t image_wo, read_write image1d_buffer_t image_rw)
{
    write_imagef(image_wo, 0, (float4)(0,0,0,0));
    write_imagei(image_wo, 0, (int4)(0,0,0,0));
    write_imagef(image_rw, 0, (float4)(0,0,0,0));
    write_imagei(image_rw, 0, (int4)(0,0,0,0));
}

// CHECK-SPIRV: FunctionParameter [[IMG1D_BUFFER_WO_TY]] [[IMG1D_BUFFER_WO:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[IMG1D_BUFFER_RW_TY]] [[IMG1D_BUFFER_RW:[0-9]+]]

// CHECK-SPIRV: ImageWrite [[IMG1D_BUFFER_WO]]
// CHECK-SPIRV: ImageWrite [[IMG1D_BUFFER_WO]]
// CHECK-SPIRV: ImageWrite [[IMG1D_BUFFER_RW]]
// CHECK-SPIRV: ImageWrite [[IMG1D_BUFFER_RW]]

// CHECK-LLVM: call spir_func void @_Z12write_imagef21ocl_image1d_buffer_woiDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei21ocl_image1d_buffer_woiDv4_i
// CHECK-LLVM: call spir_func void @_Z12write_imagef21ocl_image1d_buffer_rwiDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei21ocl_image1d_buffer_rwiDv4_i

// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_5_0_0_0_0_0_1iDv4_f
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_5_0_0_0_0_0_1iDv4_i
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_5_0_0_0_0_0_2iDv4_f
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_5_0_0_0_0_0_2iDv4_i

kernel void test_img1d_array(write_only image1d_array_t image_wo, read_write image1d_array_t image_rw)
{
    write_imagef(image_wo, (int2)(0,0), (float4)(0,0,0,0));
    write_imagei(image_wo, (int2)(0,0), (int4)(0,0,0,0));
    write_imagef(image_rw, (int2)(0,0), (float4)(0,0,0,0));
    write_imagei(image_rw, (int2)(0,0), (int4)(0,0,0,0));
    
    // LOD
    write_imagef(image_wo, (int2)(0,0), 0, (float4)(0,0,0,0));
    write_imagei(image_wo, (int2)(0,0), 0, (int4)(0,0,0,0));
}

// CHECK-SPIRV: FunctionParameter [[IMG1D_ARRAY_WO_TY]] [[IMG1D_ARRAY_WO:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[IMG1D_ARRAY_RW_TY]] [[IMG1D_ARRAY_RW:[0-9]+]]

// CHECK-SPIRV: ImageWrite [[IMG1D_ARRAY_WO]]
// CHECK-SPIRV: ImageWrite [[IMG1D_ARRAY_WO]]
// CHECK-SPIRV: ImageWrite [[IMG1D_ARRAY_RW]]
// CHECK-SPIRV: ImageWrite [[IMG1D_ARRAY_RW]]
// CHECK-SPIRV: ImageWrite [[IMG1D_ARRAY_WO]]
// CHECK-SPIRV: ImageWrite [[IMG1D_ARRAY_WO]]

// CHECK-LLVM: call spir_func void @_Z12write_imagef20ocl_image1d_array_woDv2_iDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei20ocl_image1d_array_woDv2_iDv4_i
// CHECK-LLVM: call spir_func void @_Z12write_imagef20ocl_image1d_array_rwDv2_iDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei20ocl_image1d_array_rwDv2_iDv4_i
// CHECK-LLVM: call spir_func void @_Z12write_imagef20ocl_image1d_array_woDv2_iiDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei20ocl_image1d_array_woDv2_iiDv4_i

// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_0_0_1_0_0_0_1Dv2_iDv4_f
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_0_0_1_0_0_0_1Dv2_iDv4_i
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_0_0_1_0_0_0_2Dv2_iDv4_f
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_0_0_1_0_0_0_2Dv2_iDv4_i
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_0_0_1_0_0_0_1Dv2_iDv4_fii
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_0_0_1_0_0_0_1Dv2_iDv4_iii

kernel void test_img2d_depth(write_only image2d_depth_t image_wo)
{
    write_imagef(image_wo, (int2)(0,0), (float)(0));
    write_imagef(image_wo, (int2)(0,0), (float)(0));
    
    // LOD
    write_imagef(image_wo, (int2)(0,0), 0, (float)(0));
}

// CHECK-SPIRV: FunctionParameter [[IMG2D_DEPTH_WO_TY]] [[IMG2D_DEPTH_WO:[0-9]+]]

// CHECK-SPIRV: ImageWrite [[IMG2D_DEPTH_WO]]
// CHECK-SPIRV: ImageWrite [[IMG2D_DEPTH_WO]]
// CHECK-SPIRV: ImageWrite [[IMG2D_DEPTH_WO]]

// CHECK-LLVM: call spir_func void @_Z12write_imagef20ocl_image2d_depth_woDv2_if
// CHECK-LLVM: call spir_func void @_Z12write_imagef20ocl_image2d_depth_woDv2_if
// CHECK-LLVM: call spir_func void @_Z12write_imagef20ocl_image2d_depth_woDv2_iif

// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_1_0_0_0_0_1Dv2_if
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_1_0_0_0_0_1Dv2_if
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_1_0_0_0_0_1Dv2_ifii

kernel void test_img2d_array_depth(write_only image2d_array_depth_t image_wo)
{
    write_imagef(image_wo, (int4)(0,0,0,0), (float)(0));
    write_imagef(image_wo, (int4)(0,0,0,0), (float)(0));
    
    // LOD
    write_imagef(image_wo, (int4)(0,0,0,0), 0, (float)(0));
}

// CHECK-SPIRV: FunctionParameter [[IMG2D_ARRAY_DEPTH_WO_TY]] [[IMG2D_ARRAY_DEPTH_WO:[0-9]+]]

// CHECK-SPIRV: ImageWrite [[IMG2D_ARRAY_DEPTH_WO]]
// CHECK-SPIRV: ImageWrite [[IMG2D_ARRAY_DEPTH_WO]]
// CHECK-SPIRV: ImageWrite [[IMG2D_ARRAY_DEPTH_WO]]

// CHECK-LLVM: call spir_func void @_Z12write_imagef26ocl_image2d_array_depth_woDv4_if
// CHECK-LLVM: call spir_func void @_Z12write_imagef26ocl_image2d_array_depth_woDv4_if
// CHECK-LLVM: call spir_func void @_Z12write_imagef26ocl_image2d_array_depth_woDv4_iif

// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_1_1_0_0_0_1Dv4_if
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_1_1_0_0_0_1Dv4_if
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_1_1_1_0_0_0_1Dv4_ifii

kernel void test_img3d(write_only image3d_t image_wo, read_write image3d_t image_rw)
{
    write_imagef(image_wo, (int4)(0,0,0,0), (float4)(0,0,0,0));
    write_imagei(image_wo, (int4)(0,0,0,0), (int4)(0,0,0,0));
    write_imagef(image_rw, (int4)(0,0,0,0), (float4)(0,0,0,0));
    write_imagei(image_rw, (int4)(0,0,0,0), (int4)(0,0,0,0));
    
    // LOD
    write_imagef(image_wo, (int4)(0,0,0,0), 0, (float4)(0,0,0,0));
    write_imagei(image_wo, (int4)(0,0,0,0), 0, (int4)(0,0,0,0));
}

// CHECK-SPIRV: FunctionParameter [[IMG3D_WO_TY]] [[IMG3D_WO:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[IMG3D_RW_TY]] [[IMG3D_RW:[0-9]+]]

// CHECK-SPIRV: ImageWrite [[IMG3D_WO]]
// CHECK-SPIRV: ImageWrite [[IMG3D_WO]]
// CHECK-SPIRV: ImageWrite [[IMG3D_RW]]
// CHECK-SPIRV: ImageWrite [[IMG3D_RW]]
// CHECK-SPIRV: ImageWrite [[IMG3D_WO]]
// CHECK-SPIRV: ImageWrite [[IMG3D_WO]]

// CHECK-LLVM: call spir_func void @_Z12write_imagef14ocl_image3d_woDv4_iDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei14ocl_image3d_woDv4_iS0_
// CHECK-LLVM: call spir_func void @_Z12write_imagef14ocl_image3d_rwDv4_iDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei14ocl_image3d_rwDv4_iS0_
// CHECK-LLVM: call spir_func void @_Z12write_imagef14ocl_image3d_woDv4_iiDv4_f
// CHECK-LLVM: call spir_func void @_Z12write_imagei14ocl_image3d_woDv4_iiS0_

// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_2_0_0_0_0_0_1Dv4_iDv4_f
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_2_0_0_0_0_0_1Dv4_iS2_
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_2_0_0_0_0_0_2Dv4_iDv4_f
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_2_0_0_0_0_0_2Dv4_iS2_
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_2_0_0_0_0_0_1Dv4_iDv4_fii
// CHECK-SPV-IR: call spir_func void @_Z18__spirv_ImageWritePU3AS133__spirv_Image__void_2_0_0_0_0_0_1Dv4_iS2_ii
