//==---------------- mandelbrot.cpp  - DPC++ ESIMD on-device test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// TODO enable on Windows and Level Zero
// REQUIRES: linux && gpu && opencl
// RUN: %clangxx-esimd -fsycl %s -I%S/.. -o %t.out
// RUN: %ESIMD_RUN_PLACEHOLDER %t.out %S/output.ppm %S/golden_hw.ppm

#include "esimd_test_utils.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <array>
#include <iostream>
#include <memory>

using namespace cl::sycl;
using namespace sycl::INTEL::gpu;

#ifdef _SIM_MODE_
#define CRUNCH 32
#else
#define CRUNCH 512 // emu/hw modes
#endif

#define SCALE 0.004
#define XOFF -2.09798
#define YOFF -1.19798

#define WIDTH 800
#define HEIGHT 602

template <typename ACC>
ESIMD_INLINE void mandelbrot(ACC out_image, int ix, int iy, int crunch,
                             float xOff, float yOff, float scale) {
  ix *= 8;
  iy *= 2;

  simd<int, 16> m = 0;

  // SIMT_BEGIN(16, lane) {
  for (auto lane = 0; lane < 16; ++lane) {
    int ix_lane = ix + (lane & 0x7);
    int iy_lane = iy + (lane >> 3);
    float xPos = ix_lane * scale + xOff;
    float yPos = iy_lane * scale + yOff;
    float x = 0.0f;
    float y = 0.0f;
    float xx = 0.0f;
    float yy = 0.0f;

    int mtemp = 0;
    do {
      y = x * y * 2.0f + yPos;
      x = xx - yy + xPos;
      yy = y * y;
      xx = x * x;
      mtemp += 1;
    } while ((mtemp < crunch) & (xx + yy < 4.0f));

    m.select<1, 0>(lane) = mtemp;

  } // SIMT_END

  simd<int, 16> color = (((m * 15) & 0xff)) + (((m * 7) & 0xff) * 256) +
                        (((m * 3) & 0xff) * 65536);

  // because the output is a y-tile 2D surface
  // we can only write 32-byte wide
  media_block_store<unsigned char, 2, 32>(out_image, ix * sizeof(int), iy,
                                          color.format<unsigned char>());
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: mandelbrot.exe output_file ref_file" << std::endl;
    exit(1);
  }
  // Gets the width and height of the input image.
  const unsigned img_size = WIDTH * HEIGHT * 4;
  // Sets output to blank image.
  unsigned char *buf = new unsigned char[img_size];

  {
    cl::sycl::image<2> imgOutput((unsigned int *)buf, image_channel_order::rgba,
                                 image_channel_type::unsigned_int8,
                                 range<2>{WIDTH, HEIGHT});

    // We need that many workitems
    uint range_width = WIDTH / 8;
    uint range_height = HEIGHT / 2;
    cl::sycl::range<2> GlobalRange{range_width, range_height};

    // Number of workitems in a workgroup
    cl::sycl::range<2> LocalRange{1, 1};

    queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

    auto dev = q.get_device();
    auto ctxt = q.get_context();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    auto e = q.submit([&](cl::sycl::handler &cgh) {
      auto accOutput =
          imgOutput.get_access<uint4, cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class Test>(
          GlobalRange * LocalRange, [=](item<2> it) SYCL_ESIMD_KERNEL {
            uint h_pos = it.get_id(0);
            uint v_pos = it.get_id(1);
            mandelbrot(accOutput, h_pos, v_pos, CRUNCH, XOFF, YOFF, SCALE);
          });
    });
    e.wait();
  }

  char *out_file = argv[1];
  FILE *dumpfile = fopen(out_file, "w");
  if (!dumpfile) {
    std::cerr << "Cannot open " << out_file << std::endl;
    return -2;
  }
  fprintf(dumpfile, "P6\x0d\x0a");
  fprintf(dumpfile, "%u %u\x0d\x0a", WIDTH, (HEIGHT - 2));
  fprintf(dumpfile, "%u\x0d\x0a", 255);
  fclose(dumpfile);
  dumpfile = fopen(out_file, "ab");
  for (int32_t i = 0; i < WIDTH * (HEIGHT - 2); ++i) {
    fwrite(&buf[i * 4], sizeof(char), 1, dumpfile);
    fwrite(&buf[i * 4 + 1], sizeof(char), 1, dumpfile);
    fwrite(&buf[i * 4 + 2], sizeof(char), 1, dumpfile);
  }
  fclose(dumpfile);

  bool passed = true;
  if (!esimd_test::cmp_binary_files<unsigned char>(out_file, argv[2], 0)) {
    std::cerr << out_file << " does not match the reference file " << argv[2]
              << std::endl;
    passed = false;
  }

  delete[] buf;

  if (passed) {
    std::cerr << "PASSED\n";
    return 0;
  } else {
    std::cerr << "FAILED\n";
    return 1;
  }
}
