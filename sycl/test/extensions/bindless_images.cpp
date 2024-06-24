// Check SPV_INTEL_bindless_images instructions are emitted correctly in a
// realistic scenario.

// RUN: %clangxx -S -emit-llvm -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

// Verify the mangled names of the kernel wrapper and image accesses contain
// the expected types.
// CHECK-LLVM: spir_kernel void @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperI10image_readEE
// CHECK-LLVM: tail call spir_func noundef <4 x float> @_Z17__spirv_ImageReadIDv4
// CHECK-LLVM: tail call spir_func noundef <4 x float> @_Z17__spirv_ImageReadIDv4
// CHECK-LLVM: tail call spir_func void @_Z18__spirv_ImageWriteI14

// RUN: %clangxx -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown %s -o %t.out
// RUN: llvm-spirv -spirv-ext=+SPV_INTEL_bindless_images %t.out -spirv-text -o %t.out.spv
// RUN: FileCheck %s --input-file %t.out.spv --check-prefix=CHECK-SPIRV

#include <iostream>
#include <sycl/sycl.hpp>

// Data type of image pixel components
// Arguments: Result, width
// CHECK-SPIRV: TypeFloat [[PIXELCOMPTYPE:[0-9]+]] 32

// Generate the appropriate `Result Type` used by `ConvertHandleToImageINTEL`
// and `ConvertHandleToSampledImageINTEL` Operand `7` here represents
// 'TypeVoid`. Must be `TypeVoid` as the type of the image is not known at
// compile time. The last operand is the access qualifier. With 0 read only and
// 1 write only. Arguments: TypeImage, Result, Sampled Type, Dim, Depth,
// Arrayed, MS, Sampled, Image Format.
// CHECK-SPIRV: TypeImage [[IMAGETYPE:[0-9]+]] 7 0 0 0 0 0 0 0

// Image pixel data type
// Arguments: Result, Component Type, Component Count
// CHECK-SPIRV: TypeVector [[PIXELTYPE:[0-9]+]] [[PIXELCOMPTYPE]] 4

// CHECK-SPIRV: TypeSampledImage [[SAMPIMAGETYPE:[0-9]+]] [[IMAGETYPE]]

// CHECK-SPIRV: TypeImage [[IMAGETYPEREAD:[0-9]+]] 7 0 0 0 0 0 0 1

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
class image_read;
int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  constexpr size_t width = 512;
  std::vector<float> out(width);
  std::vector<sycl::float4> dataIn(width);
  for (int i = 0; i < width; i++) {
    dataIn[i] = sycl::float4(i, i, i, i);
  }

  {
    image_descriptor desc({width}, 4, sycl::image_channel_type::fp32);

    sycl::ext::oneapi::experimental::bindless_image_sampler samp(
        sycl::addressing_mode::clamp,
        sycl::coordinate_normalization_mode::unnormalized,
        sycl::filtering_mode::nearest);

    image_mem imgMem1(desc, dev, ctxt);
    image_mem imgMem2(desc, dev, ctxt);

    unsampled_image_handle imgHandle1 = create_image(imgMem1, desc, dev, ctxt);

    sampled_image_handle imgHandle2 =
        create_image(imgMem2, samp, desc, dev, ctxt);

    q.ext_oneapi_copy(dataIn.data(), imgMem1.get_handle(), desc);
    q.ext_oneapi_copy(dataIn.data(), imgMem2.get_handle(), desc);
    q.wait_and_throw();

    sycl::buffer<float, 1> buf((float *)out.data(), width);
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(cgh, width);

      cgh.parallel_for<image_read>(width, [=](sycl::id<1> id) {
        sycl::float4 px1 = fetch_image<sycl::float4>(imgHandle1, int(id[0]));

        sycl::float4 px2 = sample_image<sycl::float4>(imgHandle2, float(id[0]));

        write_image(imgHandle1, int(id[0]), px1 + px2);

        outAcc[id] = px1[0] + px2[0];
      });
    });

    q.wait_and_throw();
    destroy_image_handle(imgHandle1, dev, ctxt);
  }

  for (int i = 0; i < width; i++) {
    std::cout << "Actual: " << out[i] << std::endl;
  }
  return 0;
}
