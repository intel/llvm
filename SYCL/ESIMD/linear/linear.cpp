//==---------------- linear.cpp  - DPC++ ESIMD on-device test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || rocm
// RUN: %clangxx -fsycl %s -I%S/.. -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out %S/linear_in.bmp %S/linear_gold_hw.bmp
// RUN: %GPU_RUN_PLACEHOLDER %t.out %S/linear_in.bmp %S/linear_gold_hw.bmp

#include "bitmap_helpers.h"
#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <array>
#include <iostream>

using namespace cl::sycl;

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: Linear.exe input_file ref_file" << std::endl;
    exit(1);
  }

  // Loads an input image named "image_in.bmp".
  auto input_image = sycl::intel::util::bitmap::BitMap::load(argv[1]);

  // Gets the width and height of the input image.
  unsigned int width = input_image.getWidth();
  unsigned int height = input_image.getHeight();
  unsigned int bpp = input_image.getBPP();

  // Checks the value of width, height and bpp(bits per pixel) of the image.
  // Only images in 8-bit RGB format are supported.
  // Only images with width and height a multiple of 8 are supported.
  if (width & 7 || height & 7 || bpp != 24) {
    std::cerr << "Error: Only images in 8-bit RGB format with width and "
              << "height a multiple of 8 are supported.\n";
    exit(1);
  }

  // Copies input image to output except for the data.
  auto output_image = input_image;

  // Sets image size in bytes. There are a total of width*height pixels and
  // each pixel occupies (out.getBPP()/8) bytes.
  unsigned int img_size = width * height * bpp / 8;

  // Sets output to blank image.
  output_image.setData(new unsigned char[img_size]);

  try {
    unsigned int img_width = width * bpp / (8 * sizeof(int));

    cl::sycl::image<2> imgInput(
        (unsigned int *)input_image.getData(), image_channel_order::rgba,
        image_channel_type::unsigned_int8, range<2>{img_width, height});

    cl::sycl::image<2> imgOutput(
        (unsigned int *)output_image.getData(), image_channel_order::rgba,
        image_channel_type::unsigned_int8, range<2>{img_width, height});

    // We need that many workitems
    uint range_width = width / 8;
    uint range_height = height / 6;
    cl::sycl::range<2> GlobalRange{range_width, range_height};

    // Number of workitems in a workgroup
    cl::sycl::range<2> LocalRange{1, 1};

    queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

    auto dev = q.get_device();
    auto ctxt = q.get_context();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    auto e = q.submit([&](cl::sycl::handler &cgh) {
      auto accInput =
          imgInput.get_access<uint4, cl::sycl::access::mode::read>(cgh);
      auto accOutput =
          imgOutput.get_access<uint4, cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class Test>(
          GlobalRange * LocalRange, [=](item<2> it) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::experimental::esimd;

            simd<unsigned char, 8 * 32> vin;
            auto in = vin.bit_cast_view<unsigned char, 8, 32>();

            simd<unsigned char, 6 * 24> vout;
            auto out = vout.bit_cast_view<uchar, 6, 24>();

            simd<float, 6 * 24> vm;
            auto m = vm.bit_cast_view<float, 6, 24>();

            uint h_pos = it.get_id(0);
            uint v_pos = it.get_id(1);

            in = media_block_load<unsigned char, 8, 32>(accInput, h_pos * 24,
                                                        v_pos * 6);

            m = in.select<6, 1, 24, 1>(1, 3);
            m += in.select<6, 1, 24, 1>(0, 0);
            m += in.select<6, 1, 24, 1>(0, 3);
            m += in.select<6, 1, 24, 1>(0, 6);
            m += in.select<6, 1, 24, 1>(1, 0);
            m += in.select<6, 1, 24, 1>(1, 6);
            m += in.select<6, 1, 24, 1>(2, 0);
            m += in.select<6, 1, 24, 1>(2, 3);
            m += in.select<6, 1, 24, 1>(2, 6);
            m = m * 0.111f;

            vout = vm;

            media_block_store<unsigned char, 6, 24>(accOutput, h_pos * 24,
                                                    v_pos * 6, out);
          });
    });
    e.wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.get_cl_code();
  }

  output_image.save("linear_out.bmp");
  bool passed = sycl::intel::util::bitmap::BitMap::checkResult("linear_out.bmp",
                                                               argv[2], 5);

  if (passed) {
    std::cerr << "PASSED\n";
    return 0;
  } else {
    std::cerr << "FAILED\n";
    return 1;
  }
}
