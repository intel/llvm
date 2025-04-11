// REQUIRES: aspect-ext_intel_legacy_image

// %O0 added because of GSD-10960. Without it, IGC will fail with
// an access violation error.
// RUN: %{build} %O0 -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: cuda
// UNSUPPORTED-INTENDED: CUDA does not support info::device::image3d_max_width
// query. Bindless images should be used instead.

// The test checks that 'image' with max allowed sizes is handled correctly.

#include <iostream>
#include <sycl/accessor_image.hpp>
#include <sycl/detail/core.hpp>
using namespace sycl;

template <int Dimensions> class CopyKernel;

bool DeviceLost = false;

template <int Dimensions>
bool testND(queue &Q, size_t XSize, size_t YSize, size_t ZSize = 1) {

  static_assert(Dimensions == 2 || Dimensions == 3,
                "Only 2D and 3D images are supported.");

  if constexpr (Dimensions == 2)
    std::cout << "Starting the test with size = {" << XSize << ", " << YSize
              << "} ... ";
  else
    std::cout << "Starting the test with size = {" << XSize << ", " << YSize
              << ", " << ZSize << "} ... ";

  const size_t NumI32Elts = XSize * YSize * ZSize * 4;
  range<Dimensions> ImgRange;
  if constexpr (Dimensions == 2)
    ImgRange = range<Dimensions>{XSize, YSize};
  else
    ImgRange = range<Dimensions>{XSize, YSize, ZSize};

  // Allocate input buffer and initialize it with some values.
  uint32_t *Input = (uint32_t *)malloc(NumI32Elts * sizeof(uint32_t));
  for (int i = 0; i < NumI32Elts; i++)
    Input[i] = i;

  // calloc to ensure that the output buffer is initialized to zero.
  uint32_t *Output = (uint32_t *)calloc(NumI32Elts, sizeof(uint32_t));

  // Create the image and submit the copy kernel.
  try {
    image<Dimensions> ImgA(Input, image_channel_order::rgba,
                           image_channel_type::unsigned_int32, ImgRange);
    image<Dimensions> ImgB(Output, image_channel_order::rgba,
                           image_channel_type::unsigned_int32, ImgRange);

    Q.submit([&](handler &CGH) {
       auto AAcc = ImgA.template get_access<uint4, access::mode::read>(CGH);
       auto BAcc = ImgB.template get_access<uint4, access::mode::write>(CGH);
       CGH.parallel_for<CopyKernel<Dimensions>>(
           ImgRange, [=](id<Dimensions> Id) {
             // Use int2 for 2D and int4 for 3D images.
             if constexpr (Dimensions == 3) {
               sycl::int4 Coord(Id[0], Id[1], Id[2], 0);
               BAcc.write(Coord, AAcc.read(Coord));
             } else {
               sycl::int2 Coord(Id[0], Id[1]);
               BAcc.write(Coord, AAcc.read(Coord));
             }
           });
     }).wait();
  } catch (exception const &e) {

    if (std::string(e.what()).find("DEVICE_LOST") != std::string::npos ||
        std::string(e.what()).find("OUT_OF_HOST_MEMORY") != std::string::npos) {
      DeviceLost = true;
      std::cout << "Device lost or out of host memory" << std::endl;
    }

    std::cout << "Failed" << std::endl;
    std::cerr << "SYCL Exception caught: " << e.what();
    free(Input);
    free(Output);
    return 1;
  }

  // Check the output buffer.
  bool HasError = false;
  for (int i = 0; i < NumI32Elts; i++) {
    if (Output[i] != i) {
      HasError = true;
      break;
    }
  }

  if (!HasError) {
    std::cout << "Passed" << std::endl;
  } else {
    std::cout << "Failed" << std::endl;
  }

  free(Input);
  free(Output);
  return HasError;
}

int main() {
  queue Q;
  device Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<info::device::name>()
            << ", Driver: " << Dev.get_info<info::device::driver_version>()
            << std::endl;

  size_t MaxWidth2D = Dev.get_info<info::device::image2d_max_width>();
  size_t MaxHeight2D = Dev.get_info<info::device::image2d_max_height>();
  std::cout << "2d: Max image width: " << MaxWidth2D << std::endl;
  std::cout << "2d: Max image height: " << MaxHeight2D << std::endl;

  size_t MaxWidth3D = Dev.get_info<info::device::image3d_max_width>();
  size_t MaxHeight3D = Dev.get_info<info::device::image3d_max_height>();
  size_t MaxDepth3D = Dev.get_info<info::device::image3d_max_depth>();
  std::cout << "3d: Max image width: " << MaxWidth3D << std::endl;
  std::cout << "3d: Max image height: " << MaxHeight3D << std::endl;
  std::cout << "3d: Max image depth: " << MaxDepth3D << std::endl;

  // Using max sizes in one image may require too much memory.
  // Check them one by one.
  bool HasError = false;
  HasError |= testND<2>(Q, MaxWidth2D, 2);
  HasError |= testND<2>(Q, 2, MaxHeight2D);

  HasError |= testND<3>(Q, MaxWidth3D, 2, 3);
  HasError |= testND<3>(Q, 2, MaxHeight3D, 3);
  HasError |= testND<3>(Q, 2, 3, MaxDepth3D);

  // This test requires a significant amount of host memory.
  // It has been observed that sometimes the test may fail with
  // OUT_OF_HOST_MEMORY error, especially when run in parallel with
  // other "high-overhead" tests. Refer CMPLRLLVM-66341.
  // If that happens, ignore the failure. An alternative is to check for
  // the available host memory and skip the test if it is too low.
  // However, that approach is still susceptible to race conditions.
  if (DeviceLost) {
    std::cout << "\n\n Device lost or ran out of memory\n"
              << "Ignoring the test result\n";
    return 0;
  }

  if (HasError)
    std::cout << "Test failed." << std::endl;
  else
    std::cout << "Test passed." << std::endl;

  return HasError ? 1 : 0;
}
