// Check SPV_INTEL_bindless_images instructions are emitted correctly in a
// realistic scenario.

// RUN: %clangxx -S -emit-llvm -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-LLVM: tail call spir_func noundef <4 x float> @_Z17__spirv_ImageReadIDv4
// CHECK-LLVM: tail call spir_func noundef <4 x float> @_Z17__spirv_ImageReadIDv4
// CHECK-LLVM: tail call spir_func void @_Z18__spirv_ImageWriteI14

// RUN: %clangxx -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown %s -o %t.out
// RUN: llvm-spirv -spirv-ext=+SPV_INTEL_bindless_images %t.out -spirv-text -o %t.out.spv
// RUN: FileCheck %s --input-file %t.out.spv --check-prefix=CHECK-SPIRV

#include <iostream>
#include <sycl/sycl.hpp>

// Determine the id of `TypeVoid`
// CHECK-SPIRV: TypeVoid [[VOIDTYPE:[0-9]+]]

// Generate the appropriate `Result Type` used by `ConvertHandleToImageINTEL`
// and `ConvertHandleToSampledImageINTEL` Must have `TypeVoid` as the type of
// the image is not known at compile time. The last operand is the access
// qualifier. With 0 read only and 1 write only. Arguments: TypeImage, Result,
// Sampled Type, Dim, Depth, Arrayed, MS, Sampled, Image Format.
// CHECK-SPIRV: TypeImage [[IMAGETYPE:[0-9]+]] [[VOIDTYPE]] 0 0 0 0 0 0 0

// Data type of image pixel components
// Arguments: Result, width
// CHECK-SPIRV: TypeFloat [[PIXELCOMPTYPE:[0-9]+]] 32

// Image pixel data type
// Arguments: Result, Component Type, Component Count
// CHECK-SPIRV: TypeVector [[PIXELTYPE:[0-9]+]] [[PIXELCOMPTYPE]] 4

// CHECK-SPIRV: TypeSampledImage [[SAMPIMAGETYPE:[0-9]+]] [[IMAGETYPE]]

// CHECK-SPIRV: TypeImage [[IMAGETYPEREAD:[0-9]+]] [[VOIDTYPE]] 0 0 0 0 0 0 1

// Convert handle to SPIR-V image
// Arguments: Result Type, Result, Handle
// CHECK-SPIRV: ConvertHandleToImageINTEL [[IMAGETYPE]] [[IMAGEVARONE:[0-9]+]] {{[0-9]+}}

// Read image
// Arguments: Result Type, Result, Image, Coords
// CHECK-SPIRV-NEXT: ImageRead [[PIXELTYPE]] {{[0-9]+}} [[IMAGEVARONE]] {{[0-9]+}}

// Convert handle to SPIR-V sampled image
// Arguments: Result Type, Result, Handle
// CHECK-SPIRV: ConvertHandleToSampledImageINTEL [[SAMPIMAGETYPE]] [[SAMPIMAGEVAR:[0-9]+]] {{[0-9]+}}

// Read sampled image
// Arguments: Result Type, Result, Image, Coords
// CHECK-SPIRV-NEXT: ImageRead [[PIXELTYPE]] {{[0-9]+}} [[SAMPIMAGEVAR]] {{[0-9]+}}

// Convert handle to SPIR-V image
// Arguments: Result Type, Result, Handle
// CHECK-SPIRV: ConvertHandleToImageINTEL [[IMAGETYPEREAD]] [[IMAGEVARTWO:[0-9]+]] {{[0-9]+}}

// Write unsampled image
// Arguments: Image, Coords, Data
// CHECK-SPIRV: ImageWrite [[IMAGEVARTWO]] {{[0-9]+}} {{[0-9]+}}

using namespace sycl::ext::oneapi::experimental;

SYCL_EXTERNAL void
image_read(sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle1,
           sycl::ext::oneapi::experimental::sampled_image_handle imgHandle2,
           sycl::accessor<float, 1, sycl::access_mode::write> outAcc,
           sycl::id<1> id) {
  auto px1 = fetch_image<sycl::float4>(imgHandle1, int(id[0]));

  auto px2 = sample_image<sycl::float4>(imgHandle2, float(id[0]));

  write_image(imgHandle1, int(id[0]), px1 + px2);

  outAcc[id] = px1[0] + px2[0];
}
