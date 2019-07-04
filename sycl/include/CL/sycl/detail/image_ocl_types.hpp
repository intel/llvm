//===-- Image_ocl_types.hpp - Image OpenCL types --------- ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file is to declare the structs with type as appropriate opencl image
// types based on Dims, AccessMode and AccessTarget.
// The macros essentially expand to -
// template <>
// struct opencl_image_type<1, access::mode::read, access::target::image> {
//   using type = __ocl_image1d_ro_t;
// };
// 
// template <>
// struct opencl_image_type<1, access::mode::write, access::target::image> {
//   using type = __ocl_image1d_array_wo_t;
// };
// 
// As an example, this can be
// used as below:
// detail::opencl_image_type<2, access::mode::read, access::target::image>::type
//    MyImage;
//

namespace cl {
namespace sycl {
namespace detail {
template <int Dimensions, access::mode AccessMode, access::target AccessTarget>
struct opencl_image_type;

#define IMAGETY_DEFINE(Dim, AccessMode, AMSuffix, Target, Ifarray_)            \
  template <>                                                                  \
  struct opencl_image_type<Dim, access::mode::AccessMode,                      \
                           access::target::Target> {                           \
    using type = __ocl_image##Dim##d_##Ifarray_##AMSuffix##_t;                 \
  };

#define IMAGETY_READ_3_DIM_IMAGE                                               \
  IMAGETY_DEFINE(1, read, ro, image, )                                         \
  IMAGETY_DEFINE(2, read, ro, image, )                                         \
  IMAGETY_DEFINE(3, read, ro, image, )

#define IMAGETY_WRITE_3_DIM_IMAGE                                              \
  IMAGETY_DEFINE(1, write, wo, image, )                                        \
  IMAGETY_DEFINE(2, write, wo, image, )                                        \
  IMAGETY_DEFINE(3, write, wo, image, )

#define IMAGETY_READ_2_DIM_IARRAY                                              \
  IMAGETY_DEFINE(1, read, ro, image_array, array_)                             \
  IMAGETY_DEFINE(2, read, ro, image_array, array_)

#define IMAGETY_WRITE_2_DIM_IARRAY                                             \
  IMAGETY_DEFINE(1, write, wo, image_array, array_)                            \
  IMAGETY_DEFINE(2, write, wo, image_array, array_)

IMAGETY_READ_3_DIM_IMAGE
IMAGETY_WRITE_3_DIM_IMAGE

IMAGETY_READ_2_DIM_IARRAY
IMAGETY_WRITE_2_DIM_IARRAY

} // namespace detail
} // namespace sycl
} // namespace cl
