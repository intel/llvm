// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// UNSUPPORTED: cuda || hip || (windows && opencl && gpu) || gpu-intel-pvc
// CUDA does not support info::device::image3d_max_width query.
// TODO: Irregular runtime fails on Windows/opencl:gpu require analysis.

// The test checks that 'image' with max allowed sizes is handled correctly.

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

void init(uint32_t *A, uint32_t *B, size_t NumI32Elts) {
  for (int I = 0; I < NumI32Elts; I++) {
    A[I] = I;
    B[I] = 0;
  }
}

int check(uint32_t *B, size_t NumI32Elts) {
  for (int I = 0; I < NumI32Elts; I++) {
    if (B[I] != I) {
      std::cout << "Failed" << std::endl;
      std::cerr << "Error for the index: " << I << ", computed: " << B[I]
                << std::endl;
      return 1;
    }
  }
  std::cout << "Passed" << std::endl;
  return 0;
}

int test2D(queue &Q, size_t XSize, size_t YSize) {
  std::cout << "Starting the test with size = {" << XSize << ", " << YSize
            << "} ... ";
  size_t NumI32Elts = XSize * YSize * 4;
  uint32_t *A = (uint32_t *)malloc(NumI32Elts * sizeof(uint32_t));
  uint32_t *B = (uint32_t *)malloc(NumI32Elts * sizeof(uint32_t));
  init(A, B, NumI32Elts);

  try {
    image<2> ImgA(A, image_channel_order::rgba,
                  image_channel_type::unsigned_int32, range<2>{XSize, YSize});
    image<2> ImgB(B, image_channel_order::rgba,
                  image_channel_type::unsigned_int32, range<2>{XSize, YSize});

    Q.submit([&](handler &CGH) {
       auto AAcc = ImgA.get_access<uint4, access::mode::read>(CGH);
       auto BAcc = ImgB.get_access<uint4, access::mode::write>(CGH);
       CGH.parallel_for<class I2D>(range<2>{XSize, YSize}, [=](id<2> Id) {
         sycl::int2 Coord(Id[0], Id[1]);
         BAcc.write(Coord, AAcc.read(Coord));
       });
     }).wait();
  } catch (exception const &e) {
    std::cout << "Failed" << std::endl;
    std::cerr << "SYCL Exception caught: " << e.what();
    return 1;
  }

  int NumErrors = check(B, NumI32Elts);
  free(A);
  free(B);
  return NumErrors;
}

int test3D(queue &Q, size_t XSize, size_t YSize, size_t ZSize) {
  std::cout << "Starting the test with size = {" << XSize << ", " << YSize
            << ", " << ZSize << "} ... ";
  size_t NumI32Elts = XSize * YSize * ZSize * 4;
  uint32_t *A = (uint32_t *)malloc(NumI32Elts * sizeof(uint32_t));
  uint32_t *B = (uint32_t *)malloc(NumI32Elts * sizeof(uint32_t));
  init(A, B, NumI32Elts);

  try {
    image<3> ImgA(A, image_channel_order::rgba,
                  image_channel_type::unsigned_int32,
                  range<3>{XSize, YSize, ZSize});
    image<3> ImgB(B, image_channel_order::rgba,
                  image_channel_type::unsigned_int32,
                  range<3>{XSize, YSize, ZSize});

    Q.submit([&](handler &CGH) {
       auto AAcc = ImgA.get_access<uint4, access::mode::read>(CGH);
       auto BAcc = ImgB.get_access<uint4, access::mode::write>(CGH);
       CGH.parallel_for<class I3D>(range<3>{XSize, YSize, ZSize},
                                   [=](id<3> Id) {
                                     sycl::int4 Coord(Id[0], Id[1], Id[2], 0);
                                     BAcc.write(Coord, AAcc.read(Coord));
                                   });
     }).wait();
  } catch (exception const &e) {
    std::cout << "Failed" << std::endl;
    std::cerr << "SYCL Exception caught: " << e.what();
    return 1;
  }

  int NumErrors = check(B, NumI32Elts);
  free(A);
  free(B);
  return NumErrors;
}

int main() {
  int NumErrors = 0;

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
  NumErrors += test2D(Q, MaxWidth2D, 2);
  NumErrors += test2D(Q, 2, MaxHeight2D);

  NumErrors += test3D(Q, MaxWidth3D, 2, 3);
  NumErrors += test3D(Q, 2, MaxHeight3D, 3);
  NumErrors += test3D(Q, 2, 3, MaxDepth3D);

  if (NumErrors)
    std::cerr << "Test failed." << std::endl;
  else
    std::cout << "Test passed." << std::endl;

  return NumErrors;
}
