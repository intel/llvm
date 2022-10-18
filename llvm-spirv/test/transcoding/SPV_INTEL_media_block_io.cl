// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv --spirv-ext=+SPV_INTEL_media_block_io %t.bc -o %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-SPV-IR

uchar __attribute__((overloadable)) intel_sub_group_media_block_read_uc(int2 src_offset, int width, int height, read_only image2d_t image);
uchar2 __attribute__((overloadable)) intel_sub_group_media_block_read_uc2(int2 src_offset, int width, int height, read_only image2d_t image);
uchar4 __attribute__((overloadable)) intel_sub_group_media_block_read_uc4(int2 src_offset, int width, int height, read_only image2d_t image);
uchar8 __attribute__((overloadable)) intel_sub_group_media_block_read_uc8(int2 src_offset, int width, int height, read_only image2d_t image);
uchar16 __attribute__((overloadable)) intel_sub_group_media_block_read_uc16(int2 src_offset, int width, int height, read_only image2d_t image);

ushort __attribute__((overloadable)) intel_sub_group_media_block_read_us(int2 src_offset, int width, int height, read_only image2d_t image);
ushort2 __attribute__((overloadable)) intel_sub_group_media_block_read_us2(int2 src_offset, int width, int height, read_only image2d_t image);
ushort4 __attribute__((overloadable)) intel_sub_group_media_block_read_us4(int2 src_offset, int width, int height, read_only image2d_t image);
ushort8 __attribute__((overloadable)) intel_sub_group_media_block_read_us8(int2 src_offset, int width, int height, read_only image2d_t image);
ushort16 __attribute__((overloadable)) intel_sub_group_media_block_read_us16(int2 src_offset, int width, int height, read_only image2d_t image);

uint __attribute__((overloadable)) intel_sub_group_media_block_read_ui(int2 src_offset, int width, int height, read_only image2d_t image);
uint2 __attribute__((overloadable)) intel_sub_group_media_block_read_ui2(int2 src_offset, int width, int height, read_only image2d_t image);
uint4 __attribute__((overloadable)) intel_sub_group_media_block_read_ui4(int2 src_offset, int width, int height, read_only image2d_t image);
uint8 __attribute__((overloadable)) intel_sub_group_media_block_read_ui8(int2 src_offset, int width, int height, read_only image2d_t image);

uchar __attribute__((overloadable)) intel_sub_group_media_block_read_uc(int2 src_offset, int width, int height, read_write image2d_t image);
uchar2 __attribute__((overloadable)) intel_sub_group_media_block_read_uc2(int2 src_offset, int width, int height, read_write image2d_t image);
uchar4 __attribute__((overloadable)) intel_sub_group_media_block_read_uc4(int2 src_offset, int width, int height, read_write image2d_t image);
uchar8 __attribute__((overloadable)) intel_sub_group_media_block_read_uc8(int2 src_offset, int width, int height, read_write image2d_t image);
uchar16 __attribute__((overloadable)) intel_sub_group_media_block_read_uc16(int2 src_offset, int width, int height, read_write image2d_t image);

ushort __attribute__((overloadable)) intel_sub_group_media_block_read_us(int2 src_offset, int width, int height, read_write image2d_t image);
ushort2 __attribute__((overloadable)) intel_sub_group_media_block_read_us2(int2 src_offset, int width, int height, read_write image2d_t image);
ushort4 __attribute__((overloadable)) intel_sub_group_media_block_read_us4(int2 src_offset, int width, int height, read_write image2d_t image);
ushort8 __attribute__((overloadable)) intel_sub_group_media_block_read_us8(int2 src_offset, int width, int height, read_write image2d_t image);
ushort16 __attribute__((overloadable)) intel_sub_group_media_block_read_us16(int2 src_offset, int width, int height, read_write image2d_t image);

uint __attribute__((overloadable)) intel_sub_group_media_block_read_ui(int2 src_offset, int width, int height, read_write image2d_t image);
uint2 __attribute__((overloadable)) intel_sub_group_media_block_read_ui2(int2 src_offset, int width, int height, read_write image2d_t image);
uint4 __attribute__((overloadable)) intel_sub_group_media_block_read_ui4(int2 src_offset, int width, int height, read_write image2d_t image);
uint8 __attribute__((overloadable)) intel_sub_group_media_block_read_ui8(int2 src_offset, int width, int height, read_write image2d_t image);

void __attribute__((overloadable)) intel_sub_group_media_block_write_uc(int2 src_offset, int width, int height, uchar pixels, write_only image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_uc2(int2 src_offset, int width, int height, uchar2 pixels, write_only image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_uc4(int2 src_offset, int width, int height, uchar4 pixels, write_only image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_uc8(int2 src_offset, int width, int height, uchar8 pixels, write_only image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_uc16(int2 src_offset, int width, int height, uchar16 pixels, write_only image2d_t image);

void __attribute__((overloadable)) intel_sub_group_media_block_write_us(int2 src_offset, int width, int height, ushort pixels, write_only image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_us2(int2 src_offset, int width, int height, ushort2 pixels, write_only image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_us4(int2 src_offset, int width, int height, ushort4 pixels, write_only image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_us8(int2 src_offset, int width, int height, ushort8 pixels, write_only image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_us16(int2 src_offset, int width, int height, ushort16 pixels, write_only image2d_t image);

void __attribute__((overloadable)) intel_sub_group_media_block_write_ui(int2 src_offset, int width, int height, uint pixels, write_only image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_ui2(int2 src_offset, int width, int height, uint2 pixels, write_only image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_ui4(int2 src_offset, int width, int height, uint4 pixels, write_only image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_ui8(int2 src_offset, int width, int height, uint8 pixels, write_only image2d_t image);

void __attribute__((overloadable)) intel_sub_group_media_block_write_uc(int2 src_offset, int width, int height, uchar pixels, read_write image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_uc2(int2 src_offset, int width, int height, uchar2 pixels, read_write image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_uc4(int2 src_offset, int width, int height, uchar4 pixels, read_write image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_uc8(int2 src_offset, int width, int height, uchar8 pixels, read_write image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_uc16(int2 src_offset, int width, int height, uchar16 pixels, read_write image2d_t image);

void __attribute__((overloadable)) intel_sub_group_media_block_write_us(int2 src_offset, int width, int height, ushort pixels, read_write image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_us2(int2 src_offset, int width, int height, ushort2 pixels, read_write image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_us4(int2 src_offset, int width, int height, ushort4 pixels, read_write image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_us8(int2 src_offset, int width, int height, ushort8 pixels, read_write image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_us16(int2 src_offset, int width, int height, ushort16 pixels, read_write image2d_t image);

void __attribute__((overloadable)) intel_sub_group_media_block_write_ui(int2 src_offset, int width, int height, uint pixels, read_write image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_ui2(int2 src_offset, int width, int height, uint2 pixels, read_write image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_ui4(int2 src_offset, int width, int height, uint4 pixels, read_write image2d_t image);
void __attribute__((overloadable)) intel_sub_group_media_block_write_ui8(int2 src_offset, int width, int height, uint8 pixels, read_write image2d_t image);

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
__kernel void intel_media_block_test(int2 edgeCoord, __read_only image2d_t src_luma_image,
                         __write_only image2d_t dst_luma_image) {
  // Byte sized read operations
  uchar uc =
      intel_sub_group_media_block_read_uc(edgeCoord, 1, 16, src_luma_image);
  uchar2 uc2 =
      intel_sub_group_media_block_read_uc2(edgeCoord, 1, 16, src_luma_image);
  uchar4 uc4 =
      intel_sub_group_media_block_read_uc4(edgeCoord, 1, 16, src_luma_image);
  uchar8 uc8 =
      intel_sub_group_media_block_read_uc8(edgeCoord, 1, 16, src_luma_image);
  uchar16 uc16 =
      intel_sub_group_media_block_read_uc16(edgeCoord, 1, 16, src_luma_image);

  // Word sized read operations
  ushort us =
      intel_sub_group_media_block_read_us(edgeCoord, 1, 16, src_luma_image);
  ushort2 us2 =
      intel_sub_group_media_block_read_us2(edgeCoord, 1, 16, src_luma_image);
  ushort4 us4 =
      intel_sub_group_media_block_read_us4(edgeCoord, 1, 16, src_luma_image);
  ushort8 us8 =
      intel_sub_group_media_block_read_us8(edgeCoord, 1, 16, src_luma_image);
  ushort16 us16 =
      intel_sub_group_media_block_read_us16(edgeCoord, 1, 16, src_luma_image);

  // Double Word (DWORD) sized read operations
  uint ui =
      intel_sub_group_media_block_read_ui(edgeCoord, 1, 16, src_luma_image);
  uint2 ui2 =
      intel_sub_group_media_block_read_ui2(edgeCoord, 1, 16, src_luma_image);
  uint4 ui4 =
      intel_sub_group_media_block_read_ui4(edgeCoord, 1, 16, src_luma_image);
  uint8 ui8 =
      intel_sub_group_media_block_read_ui8(edgeCoord, 1, 16, src_luma_image);

  // Byte sized write operations
  intel_sub_group_media_block_write_uc(edgeCoord, 1, 16, uc, dst_luma_image);
  intel_sub_group_media_block_write_uc2(edgeCoord, 1, 16, uc2, dst_luma_image);
  intel_sub_group_media_block_write_uc4(edgeCoord, 1, 16, uc4, dst_luma_image);
  intel_sub_group_media_block_write_uc8(edgeCoord, 1, 16, uc8, dst_luma_image);
  intel_sub_group_media_block_write_uc16(edgeCoord, 1, 16, uc16, dst_luma_image);

  // Word sized write operations
  intel_sub_group_media_block_write_us(edgeCoord, 1, 16, us, dst_luma_image);
  intel_sub_group_media_block_write_us2(edgeCoord, 1, 16, us2, dst_luma_image);
  intel_sub_group_media_block_write_us4(edgeCoord, 1, 16, us4, dst_luma_image);
  intel_sub_group_media_block_write_us8(edgeCoord, 1, 16, us8, dst_luma_image);
  intel_sub_group_media_block_write_us16(edgeCoord, 1, 16, us16, dst_luma_image);

  // Double word (DWORD) sized write operations
  intel_sub_group_media_block_write_ui(edgeCoord, 1, 16, ui, dst_luma_image);
  intel_sub_group_media_block_write_ui2(edgeCoord, 1, 16, ui2, dst_luma_image);
  intel_sub_group_media_block_write_ui4(edgeCoord, 1, 16, ui4, dst_luma_image);
  intel_sub_group_media_block_write_ui8(edgeCoord, 1, 16, ui8, dst_luma_image);
}

// CHECK-SPIRV: Capability SubgroupImageMediaBlockIOINTEL
// CHECK-SPIRV: Extension "SPV_INTEL_media_block_io"

// CHECK-SPIRV: TypeInt [[TypeInt:[0-9]+]] 32
// CHECK-SPIRV: TypeInt [[TypeChar:[0-9]+]] 8
// CHECK-SPIRV: TypeInt [[TypeShort:[0-9]+]] 16
// CHECK-SPIRV: Constant [[TypeInt]] [[One:[0-9]+]] 1
// CHECK-SPIRV: Constant [[TypeInt]] [[Sixteen:[0-9]+]] 16
// CHECK-SPIRV: TypeVector [[TypeInt2:[0-9]+]] [[TypeInt]] 2
// CHECK-SPIRV: TypeVector [[TypeChar2:[0-9]+]] [[TypeChar]] 2
// CHECK-SPIRV: TypeVector [[TypeChar4:[0-9]+]] [[TypeChar]] 4
// CHECK-SPIRV: TypeVector [[TypeChar8:[0-9]+]] [[TypeChar]] 8
// CHECK-SPIRV: TypeVector [[TypeChar16:[0-9]+]] [[TypeChar]] 16
// CHECK-SPIRV: TypeVector [[TypeShort2:[0-9]+]] [[TypeShort]] 2
// CHECK-SPIRV: TypeVector [[TypeShort4:[0-9]+]] [[TypeShort]] 4
// CHECK-SPIRV: TypeVector [[TypeShort8:[0-9]+]] [[TypeShort]] 8
// CHECK-SPIRV: TypeVector [[TypeShort16:[0-9]+]] [[TypeShort]] 16
// CHECK-SPIRV: TypeVector [[TypeInt4:[0-9]+]] [[TypeInt]] 4
// CHECK-SPIRV: TypeVector [[TypeInt8:[0-9]+]] [[TypeInt]] 8

// CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[Coord:[0-9]+]]
// CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[SrcImage:[0-9]+]]
// CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[DstImage:[0-9]+]]

// CHECK-SPIRV: SubgroupImageMediaBlockReadINTEL [[TypeChar]] [[uc:[0-9]+]] [[SrcImage]] [[Coord]] [[One]] [[Sixteen]]
// CHECK-SPIRV: SubgroupImageMediaBlockReadINTEL [[TypeChar2]] [[uc2:[0-9]+]] [[SrcImage]] [[Coord]] [[One]] [[Sixteen]]
// CHECK-SPIRV: SubgroupImageMediaBlockReadINTEL [[TypeChar4]] [[uc4:[0-9]+]] [[SrcImage]] [[Coord]] [[One]] [[Sixteen]]
// CHECK-SPIRV: SubgroupImageMediaBlockReadINTEL [[TypeChar8]] [[uc8:[0-9]+]] [[SrcImage]] [[Coord]] [[One]] [[Sixteen]]
// CHECK-SPIRV: SubgroupImageMediaBlockReadINTEL [[TypeChar16]] [[uc16:[0-9]+]] [[SrcImage]] [[Coord]] [[One]] [[Sixteen]]
// CHECK-SPIRV: SubgroupImageMediaBlockReadINTEL [[TypeShort]] [[us:[0-9]+]] [[SrcImage]] [[Coord]] [[One]] [[Sixteen]]
// CHECK-SPIRV: SubgroupImageMediaBlockReadINTEL [[TypeShort2]] [[us2:[0-9]+]] [[SrcImage]] [[Coord]] [[One]] [[Sixteen]]
// CHECK-SPIRV: SubgroupImageMediaBlockReadINTEL [[TypeShort4]] [[us4:[0-9]+]] [[SrcImage]] [[Coord]] [[One]] [[Sixteen]]
// CHECK-SPIRV: SubgroupImageMediaBlockReadINTEL [[TypeShort8]] [[us8:[0-9]+]] [[SrcImage]] [[Coord]] [[One]] [[Sixteen]]
// CHECK-SPIRV: SubgroupImageMediaBlockReadINTEL [[TypeShort16]] [[us16:[0-9]+]] [[SrcImage]] [[Coord]] [[One]] [[Sixteen]]
// CHECK-SPIRV: SubgroupImageMediaBlockReadINTEL [[TypeInt]] [[ui:[0-9]+]] [[SrcImage]] [[Coord]] [[One]] [[Sixteen]]
// CHECK-SPIRV: SubgroupImageMediaBlockReadINTEL [[TypeInt2]] [[ui2:[0-9]+]] [[SrcImage]] [[Coord]] [[One]] [[Sixteen]]
// CHECK-SPIRV: SubgroupImageMediaBlockReadINTEL [[TypeInt4]] [[ui4:[0-9]+]] [[SrcImage]] [[Coord]] [[One]] [[Sixteen]]
// CHECK-SPIRV: SubgroupImageMediaBlockReadINTEL [[TypeInt8]] [[ui8:[0-9]+]] [[SrcImage]] [[Coord]] [[One]] [[Sixteen]]
// CHECK-SPIRV: SubgroupImageMediaBlockWriteINTEL [[DstImage]] [[Coord]] [[One]] [[Sixteen]] [[uc]]
// CHECK-SPIRV: SubgroupImageMediaBlockWriteINTEL [[DstImage]] [[Coord]] [[One]] [[Sixteen]] [[uc2]]
// CHECK-SPIRV: SubgroupImageMediaBlockWriteINTEL [[DstImage]] [[Coord]] [[One]] [[Sixteen]] [[uc4]]
// CHECK-SPIRV: SubgroupImageMediaBlockWriteINTEL [[DstImage]] [[Coord]] [[One]] [[Sixteen]] [[uc8]]
// CHECK-SPIRV: SubgroupImageMediaBlockWriteINTEL [[DstImage]] [[Coord]] [[One]] [[Sixteen]] [[uc16]]
// CHECK-SPIRV: SubgroupImageMediaBlockWriteINTEL [[DstImage]] [[Coord]] [[One]] [[Sixteen]] [[us]]
// CHECK-SPIRV: SubgroupImageMediaBlockWriteINTEL [[DstImage]] [[Coord]] [[One]] [[Sixteen]] [[us2]]
// CHECK-SPIRV: SubgroupImageMediaBlockWriteINTEL [[DstImage]] [[Coord]] [[One]] [[Sixteen]] [[us4]]
// CHECK-SPIRV: SubgroupImageMediaBlockWriteINTEL [[DstImage]] [[Coord]] [[One]] [[Sixteen]] [[us8]]
// CHECK-SPIRV: SubgroupImageMediaBlockWriteINTEL [[DstImage]] [[Coord]] [[One]] [[Sixteen]] [[us16]]
// CHECK-SPIRV: SubgroupImageMediaBlockWriteINTEL [[DstImage]] [[Coord]] [[One]] [[Sixteen]] [[ui]]
// CHECK-SPIRV: SubgroupImageMediaBlockWriteINTEL [[DstImage]] [[Coord]] [[One]] [[Sixteen]] [[ui2]]
// CHECK-SPIRV: SubgroupImageMediaBlockWriteINTEL [[DstImage]] [[Coord]] [[One]] [[Sixteen]] [[ui4]]
// CHECK-SPIRV: SubgroupImageMediaBlockWriteINTEL [[DstImage]] [[Coord]] [[One]] [[Sixteen]] [[ui8]]

// CHECK-LLVM: call spir_func i8 @_Z35intel_sub_group_media_block_read_ucDv2_iii14ocl_image2d_ro(<2 x i32> %edgeCoord, i32 1, i32 16, %opencl.image2d_ro_t addrspace(1)* %src_luma_image)
// CHECK-LLVM: call spir_func <2 x i8> @_Z36intel_sub_group_media_block_read_uc2Dv2_iii14ocl_image2d_ro(<2 x i32> %edgeCoord, i32 1, i32 16, %opencl.image2d_ro_t addrspace(1)* %src_luma_image)
// CHECK-LLVM: call spir_func <4 x i8> @_Z36intel_sub_group_media_block_read_uc4Dv2_iii14ocl_image2d_ro(<2 x i32> %edgeCoord, i32 1, i32 16, %opencl.image2d_ro_t addrspace(1)* %src_luma_image)
// CHECK-LLVM: call spir_func <8 x i8> @_Z36intel_sub_group_media_block_read_uc8Dv2_iii14ocl_image2d_ro(<2 x i32> %edgeCoord, i32 1, i32 16, %opencl.image2d_ro_t addrspace(1)* %src_luma_image)
// CHECK-LLVM: call spir_func <16 x i8> @_Z37intel_sub_group_media_block_read_uc16Dv2_iii14ocl_image2d_ro(<2 x i32> %edgeCoord, i32 1, i32 16, %opencl.image2d_ro_t addrspace(1)* %src_luma_image) 
// CHECK-LLVM: call spir_func i16 @_Z35intel_sub_group_media_block_read_usDv2_iii14ocl_image2d_ro(<2 x i32> %edgeCoord, i32 1, i32 16, %opencl.image2d_ro_t addrspace(1)* %src_luma_image)
// CHECK-LLVM: call spir_func <2 x i16> @_Z36intel_sub_group_media_block_read_us2Dv2_iii14ocl_image2d_ro(<2 x i32> %edgeCoord, i32 1, i32 16, %opencl.image2d_ro_t addrspace(1)* %src_luma_image)
// CHECK-LLVM: call spir_func <4 x i16> @_Z36intel_sub_group_media_block_read_us4Dv2_iii14ocl_image2d_ro(<2 x i32> %edgeCoord, i32 1, i32 16, %opencl.image2d_ro_t addrspace(1)* %src_luma_image)
// CHECK-LLVM: call spir_func <8 x i16> @_Z36intel_sub_group_media_block_read_us8Dv2_iii14ocl_image2d_ro(<2 x i32> %edgeCoord, i32 1, i32 16, %opencl.image2d_ro_t addrspace(1)* %src_luma_image)
// CHECK-LLVM: call spir_func <16 x i16> @_Z37intel_sub_group_media_block_read_us16Dv2_iii14ocl_image2d_ro(<2 x i32> %edgeCoord, i32 1, i32 16, %opencl.image2d_ro_t addrspace(1)* %src_luma_image)
// CHECK-LLVM: call spir_func i32 @_Z35intel_sub_group_media_block_read_uiDv2_iii14ocl_image2d_ro(<2 x i32> %edgeCoord, i32 1, i32 16, %opencl.image2d_ro_t addrspace(1)* %src_luma_image)
// CHECK-LLVM: call spir_func <2 x i32> @_Z36intel_sub_group_media_block_read_ui2Dv2_iii14ocl_image2d_ro(<2 x i32> %edgeCoord, i32 1, i32 16, %opencl.image2d_ro_t addrspace(1)* %src_luma_image)
// CHECK-LLVM: call spir_func <4 x i32> @_Z36intel_sub_group_media_block_read_ui4Dv2_iii14ocl_image2d_ro(<2 x i32> %edgeCoord, i32 1, i32 16, %opencl.image2d_ro_t addrspace(1)* %src_luma_image)
// CHECK-LLVM: call spir_func <8 x i32> @_Z36intel_sub_group_media_block_read_ui8Dv2_iii14ocl_image2d_ro(<2 x i32> %edgeCoord, i32 1, i32 16, %opencl.image2d_ro_t addrspace(1)* %src_luma_image)
// CHECK-LLVM: call spir_func void @_Z36intel_sub_group_media_block_write_ucDv2_iiih14ocl_image2d_wo(<2 x i32> %edgeCoord, i32 1, i32 16, i8 %{{.*}}, %opencl.image2d_wo_t addrspace(1)* %dst_luma_image)
// CHECK-LLVM: call spir_func void @_Z37intel_sub_group_media_block_write_uc2Dv2_iiiDv2_h14ocl_image2d_wo(<2 x i32> %edgeCoord, i32 1, i32 16, <2 x i8> %{{.*}}, %opencl.image2d_wo_t addrspace(1)* %dst_luma_image)
// CHECK-LLVM: call spir_func void @_Z37intel_sub_group_media_block_write_uc4Dv2_iiiDv4_h14ocl_image2d_wo(<2 x i32> %edgeCoord, i32 1, i32 16, <4 x i8> %{{.*}}, %opencl.image2d_wo_t addrspace(1)* %dst_luma_image)
// CHECK-LLVM: call spir_func void @_Z37intel_sub_group_media_block_write_uc8Dv2_iiiDv8_h14ocl_image2d_wo(<2 x i32> %edgeCoord, i32 1, i32 16, <8 x i8> %{{.*}}, %opencl.image2d_wo_t addrspace(1)* %dst_luma_image)
// CHECK-LLVM: call spir_func void @_Z38intel_sub_group_media_block_write_uc16Dv2_iiiDv16_h14ocl_image2d_wo(<2 x i32> %edgeCoord, i32 1, i32 16, <16 x i8> %{{.*}}, %opencl.image2d_wo_t addrspace(1)* %dst_luma_image)
// CHECK-LLVM: call spir_func void @_Z36intel_sub_group_media_block_write_usDv2_iiit14ocl_image2d_wo(<2 x i32> %edgeCoord, i32 1, i32 16, i16 %{{.*}}, %opencl.image2d_wo_t addrspace(1)* %dst_luma_image)
// CHECK-LLVM: call spir_func void @_Z37intel_sub_group_media_block_write_us2Dv2_iiiDv2_t14ocl_image2d_wo(<2 x i32> %edgeCoord, i32 1, i32 16, <2 x i16> %{{.*}}, %opencl.image2d_wo_t addrspace(1)* %dst_luma_image)
// CHECK-LLVM: call spir_func void @_Z37intel_sub_group_media_block_write_us4Dv2_iiiDv4_t14ocl_image2d_wo(<2 x i32> %edgeCoord, i32 1, i32 16, <4 x i16> %{{.*}}, %opencl.image2d_wo_t addrspace(1)* %dst_luma_image)
// CHECK-LLVM: call spir_func void @_Z37intel_sub_group_media_block_write_us8Dv2_iiiDv8_t14ocl_image2d_wo(<2 x i32> %edgeCoord, i32 1, i32 16, <8 x i16> %{{.*}}, %opencl.image2d_wo_t addrspace(1)* %dst_luma_image)
// CHECK-LLVM: call spir_func void @_Z38intel_sub_group_media_block_write_us16Dv2_iiiDv16_t14ocl_image2d_wo(<2 x i32> %edgeCoord, i32 1, i32 16, <16 x i16> %{{.*}}, %opencl.image2d_wo_t addrspace(1)* %dst_luma_image)
// CHECK-LLVM: call spir_func void @_Z36intel_sub_group_media_block_write_uiDv2_iiij14ocl_image2d_wo(<2 x i32> %edgeCoord, i32 1, i32 16, i32 %{{.*}}, %opencl.image2d_wo_t addrspace(1)* %dst_luma_image)
// CHECK-LLVM: call spir_func void @_Z37intel_sub_group_media_block_write_ui2Dv2_iiiDv2_j14ocl_image2d_wo(<2 x i32> %edgeCoord, i32 1, i32 16, <2 x i32> %{{.*}}, %opencl.image2d_wo_t addrspace(1)* %dst_luma_image)
// CHECK-LLVM: call spir_func void @_Z37intel_sub_group_media_block_write_ui4Dv2_iiiDv4_j14ocl_image2d_wo(<2 x i32> %edgeCoord, i32 1, i32 16, <4 x i32> %{{.*}}, %opencl.image2d_wo_t addrspace(1)* %dst_luma_image)
// CHECK-LLVM: call spir_func void @_Z37intel_sub_group_media_block_write_ui8Dv2_iiiDv8_j14ocl_image2d_wo(<2 x i32> %edgeCoord, i32 1, i32 16, <8 x i32> %{{.*}}, %opencl.image2d_wo_t addrspace(1)* %dst_luma_image)

// CHECK-SPV-IR: call spir_func i8 @_Z46__spirv_SubgroupImageMediaBlockReadINTEL_RcharPU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %src_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16)
// CHECK-SPV-IR: call spir_func <2 x i8> @_Z47__spirv_SubgroupImageMediaBlockReadINTEL_Rchar2PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %src_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16)
// CHECK-SPV-IR: call spir_func <4 x i8> @_Z47__spirv_SubgroupImageMediaBlockReadINTEL_Rchar4PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %src_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16)
// CHECK-SPV-IR: call spir_func <8 x i8> @_Z47__spirv_SubgroupImageMediaBlockReadINTEL_Rchar8PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %src_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16)
// CHECK-SPV-IR: call spir_func <16 x i8> @_Z48__spirv_SubgroupImageMediaBlockReadINTEL_Rchar16PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %src_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16)
// CHECK-SPV-IR: call spir_func i16 @_Z47__spirv_SubgroupImageMediaBlockReadINTEL_RshortPU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %src_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16)
// CHECK-SPV-IR: call spir_func <2 x i16> @_Z48__spirv_SubgroupImageMediaBlockReadINTEL_Rshort2PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %src_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16)
// CHECK-SPV-IR: call spir_func <4 x i16> @_Z48__spirv_SubgroupImageMediaBlockReadINTEL_Rshort4PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %src_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16)
// CHECK-SPV-IR: call spir_func <8 x i16> @_Z48__spirv_SubgroupImageMediaBlockReadINTEL_Rshort8PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %src_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16)
// CHECK-SPV-IR: call spir_func <16 x i16> @_Z49__spirv_SubgroupImageMediaBlockReadINTEL_Rshort16PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %src_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16)
// CHECK-SPV-IR: call spir_func i32 @_Z45__spirv_SubgroupImageMediaBlockReadINTEL_RintPU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %src_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16)
// CHECK-SPV-IR: call spir_func <2 x i32> @_Z46__spirv_SubgroupImageMediaBlockReadINTEL_Rint2PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %src_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16)
// CHECK-SPV-IR: call spir_func <4 x i32> @_Z46__spirv_SubgroupImageMediaBlockReadINTEL_Rint4PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %src_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16)
// CHECK-SPV-IR: call spir_func <8 x i32> @_Z46__spirv_SubgroupImageMediaBlockReadINTEL_Rint8PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %src_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16)
// CHECK-SPV-IR: call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiic(%spirv.Image._void_1_0_0_0_0_0_1 addrspace(1)* %dst_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16, i8 %{{.*}})
// CHECK-SPV-IR: call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv2_c(%spirv.Image._void_1_0_0_0_0_0_1 addrspace(1)* %dst_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16, <2 x i8> %{{.*}})
// CHECK-SPV-IR: call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv4_c(%spirv.Image._void_1_0_0_0_0_0_1 addrspace(1)* %dst_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16, <4 x i8> %{{.*}})
// CHECK-SPV-IR: call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv8_c(%spirv.Image._void_1_0_0_0_0_0_1 addrspace(1)* %dst_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16, <8 x i8> %{{.*}})
// CHECK-SPV-IR: call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv16_c(%spirv.Image._void_1_0_0_0_0_0_1 addrspace(1)* %dst_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16, <16 x i8> %{{.*}})
// CHECK-SPV-IR: call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiis(%spirv.Image._void_1_0_0_0_0_0_1 addrspace(1)* %dst_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16, i16 %{{.*}})
// CHECK-SPV-IR: call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv2_s(%spirv.Image._void_1_0_0_0_0_0_1 addrspace(1)* %dst_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16, <2 x i16> %{{.*}})
// CHECK-SPV-IR: call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv4_s(%spirv.Image._void_1_0_0_0_0_0_1 addrspace(1)* %dst_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16, <4 x i16> %{{.*}})
// CHECK-SPV-IR: call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv8_s(%spirv.Image._void_1_0_0_0_0_0_1 addrspace(1)* %dst_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16, <8 x i16> %{{.*}})
// CHECK-SPV-IR: call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv16_s(%spirv.Image._void_1_0_0_0_0_0_1 addrspace(1)* %dst_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16, <16 x i16> %{{.*}})
// CHECK-SPV-IR: call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiii(%spirv.Image._void_1_0_0_0_0_0_1 addrspace(1)* %dst_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16, i32 %{{.*}})
// CHECK-SPV-IR: call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiS2_(%spirv.Image._void_1_0_0_0_0_0_1 addrspace(1)* %dst_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16, <2 x i32> %{{.*}})
// CHECK-SPV-IR: call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv4_i(%spirv.Image._void_1_0_0_0_0_0_1 addrspace(1)* %dst_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16, <4 x i32> %{{.*}})
// CHECK-SPV-IR: call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv8_i(%spirv.Image._void_1_0_0_0_0_0_1 addrspace(1)* %dst_luma_image, <2 x i32> %edgeCoord, i32 1, i32 16, <8 x i32> %{{.*}})
