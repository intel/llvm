//==--------- plugin_printers.hpp - Printers for the Plugin Interface ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Print functions used for the Plguin Interface tracing.

#pragma once

#include <CL/sycl/detail/pi.hpp>
#include <cstdio>
#include <type_traits>
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
namespace pi {

template <typename T>
inline typename std::enable_if<!std::is_pointer<T>::value, void>::type
print(T val) {
  printf("<unknown> : %s\n", std::to_string(val).c_str());
}
template <typename T>
inline typename std::enable_if<std::is_pointer<T>::value, void>::type
print(T val) {
  printf("<unknown> : %p\n", reinterpret_cast<const void *>(val));
}

template <> inline void print<>(PiPlatform val) {
  printf("pi_platform : %p\n", (void *)val);
}

template <> inline void print<>(PiEvent val) {
  printf("pi_event : %p\n", (void *)val);
}

template <> inline void print<>(PiMem val) {
  printf("pi_mem : %p\n", (void *)val);
}

template <> inline void print<>(PiEvent *val) {
  printf("pi_event * : %p", (void *)val);
  if (val)
    printf("[ %p ... ]", (void *)*val);
  else
    printf("[ nullptr ]");
  printf("\n");
}

template <> inline void print<>(const PiEvent *val) {
  printf("const pi_event * : %p", (const void *)val);
  if (val)
    printf("[ %p ... ]", (void *)*val);
  else
    printf("[ nullptr ]");
  printf("\n");
}

template <> inline void print<>(pi_buffer_region rgn) {
  printf("pi_buffer_region origin/size : %lu/%lu\n", rgn->origin, rgn->size);
}

template <> inline void print<>(pi_buff_rect_region rgn) {
  printf("pi_buff_rect_region width_bytes/height/depth : %lu/%lu/%lu\n",
         rgn->width_bytes, rgn->height_scalar, rgn->depth_scalar);
}

template <> inline void print<>(pi_buff_rect_offset off) {
  printf("pi_buff_rect_offset x_bytes/y/z : %lu/%lu/%lu\n", off->x_bytes,
         off->y_scalar, off->z_scalar);
}

template <> inline void print<>(pi_image_region rgn) {
  printf("pi_image_region width/height/depth : %lu/%lu/%lu\n", rgn->width,
         rgn->height, rgn->depth);
}

template <> inline void print<>(pi_image_offset off) {
  printf("pi_image_offset x/y/z : %lu/%lu/%lu\n", off->x, off->y, off->z);
}

template <> inline void print<>(const pi_image_desc *desc) {
  printf("image_desc w/h/d : %lu/%lu/%lu%s%lu/%lu/%lu%s%d/%d/%d\n",
         desc->image_width, desc->image_height, desc->image_depth,
         "  --  arrSz/row/slice : ", desc->image_array_size,
         desc->image_row_pitch, desc->image_slice_pitch,
         "  --  num_mip_lvls/num_smpls/image_type : ", desc->num_mip_levels,
         desc->num_samples, desc->image_type);
}

template <> inline void print<>(PiResult val) {
  printf("pi_result : ");
  if (val == PI_SUCCESS)
    printf("PI_SUCCESS\n");
  else
    printf("%d\n", val);
}

// cout does not resolve a nullptr.
template <> inline void print<>(std::nullptr_t) { printf("<nullptr>\n"); }

template <> inline void print<>(char *val) {
  printf("<char * > : %p\n", static_cast<void *>(val));
}

template <> inline void print<>(const char *val) {
  printf("<const char *>: %s\n", val);
}

inline void printArgs(void) {}
template <typename Arg0, typename... Args>
void printArgs(Arg0 arg0, Args... args) {
  printf("\t");
  print(arg0);
  pi::printArgs(std::forward<Args>(args)...);
}

template <typename T> struct printOut {
  printOut(T) {}
}; // Do nothing

template <> struct printOut<PiEvent *> {
  printOut(PiEvent *val) {
    printf("\t[out]pi_event * : %p", static_cast<void *>(val));
    if (val)
      printf("[ %p ... ]", static_cast<void *>(*val));
    else
      printf("[ nullptr ]");
    printf("\n");
  }
};

template <> struct printOut<PiMem *> {
  printOut(PiMem *val) {
    printf("\t[out]pi_mem * : %p", (void *)val);
    if (val)
      printf("[ %p ... ]", (void *)*val);
    else
      printf("[ nullptr ]");
    printf("\n");
  }
};

template <> struct printOut<void *> {
  printOut(void *val) { printf("\t[out]void * : %p\n", val); }
};

template <typename T> struct printOut<T **> {
  printOut(T **val) {
    printf("\t[out]<unknown> ** : %p", (void *)val);
    if (val)
      printf("[ %p ... ]", (const void *)*val);
    else
      printf("[ nullptr ]");
    printf("\n");
  }
};

inline void printOuts(void) {}
template <typename Arg0, typename... Args>
void printOuts(Arg0 arg0, Args... args) {
  using T = decltype(arg0);
  printOut<T> a(arg0);
  printOuts(std::forward<Args>(args)...);
}

} // namespace pi
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
