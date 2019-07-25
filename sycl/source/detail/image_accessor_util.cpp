//==----------- image_accessor_util.cpp ------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/builtins.hpp>

namespace cl {
namespace sycl {
namespace detail {

// For Nearest Filtering mode, process cl_float4 Coordinates and return the
// appropriate Pixel Coordinates based on Addressing Mode.
cl_int4 getPixelCoordNearestFiltMode(cl_float4 Coord_uvw,
                                     addressing_mode SmplAddrMode,
                                     range<3> ImgRange) {
  cl_float u = Coord_uvw.x();
  cl_float v = Coord_uvw.y();
  cl_float w = Coord_uvw.z();

  cl_int i = 0;
  cl_int j = 0;
  cl_int k = 0;
  cl_int width = ImgRange[0];
  cl_int height = ImgRange[1];
  cl_int depth = ImgRange[2];
  switch (SmplAddrMode) {
  case addressing_mode::mirrored_repeat:
    // TODO: Add the computations.
    break;
  case addressing_mode::repeat:
    // TODO: Add the computations.
    break;
  case addressing_mode::clamp_to_edge:
    i = cl::sycl::clamp((int)cl::sycl::floor(u), 0, (width - 1));
    j = cl::sycl::clamp((int)cl::sycl::floor(v), 0, (height - 1));
    k = cl::sycl::clamp((int)cl::sycl::floor(w), 0, (depth - 1));
    break;
  case addressing_mode::clamp:
    i = cl::sycl::clamp((int)cl::sycl::floor(u), -1, width);
    j = cl::sycl::clamp((int)cl::sycl::floor(v), -1, height);
    k = cl::sycl::clamp((int)cl::sycl::floor(w), -1, depth);
    break;
  case addressing_mode::none:
    i = (int)cl::sycl::floor(u);
    j = (int)cl::sycl::floor(v);
    k = (int)cl::sycl::floor(w);
    break;
  }
  return cl_int4{i, j, k, 0};
}

} // namespace detail
} // namespace sycl
} // namespace cl
