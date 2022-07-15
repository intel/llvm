//==--------- plugin_printers.hpp - Printers for the Plugin Interface ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Print functions used for the Plguin Interface tracing.

#pragma once

#include <cstdio>
#include <sycl/detail/pi.hpp>
#include <type_traits>
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
namespace pi {

inline void print_pointer(const void *p, bool newLine) {
  if (p) {
    printf("%p", p);
  } else {
    printf("0");
  }
  if (newLine) {
    printf("\n");
  }
}

template <typename T>
inline typename std::enable_if<!std::is_pointer<T>::value, void>::type
print(T val) {
  printf("<unknown> : %s\n", std::to_string(val).c_str());
}
template <typename T>
inline typename std::enable_if<std::is_pointer<T>::value, void>::type
print(T val) {
  printf("<unknown> : ");
  print_pointer(reinterpret_cast<const void *>(val), true);
}

template <> inline void print<>(PiPlatform val) {
  printf("pi_platform : ");
  print_pointer(val, true);
}

template <> inline void print<>(PiEvent val) {
  printf("pi_event : ");
  print_pointer(val, true);
}

template <> inline void print<>(PiMem val) {
  printf("pi_mem : ");
  print_pointer(val, true);
}

template <> inline void print<>(PiEvent *val) {
  printf("pi_event * : ");
  print_pointer(val, false);
  if (val) {
    printf("[ ");
    print_pointer(*val, false);
    printf(" ... ]");
  } else {
    printf("[ nullptr ]");
  }
  printf("\n");
}

template <> inline void print<>(const PiEvent *val) {
  printf("const pi_event * : ");
  print_pointer(val, false);
  if (val) {
    printf("[ ");
    print_pointer(*val, false);
    printf(" ... ]");
  } else {
    printf("[ nullptr ]");
  }
  printf("\n");
}

template <> inline void print<>(pi_buffer_region rgn) {
  printf("pi_buffer_region origin/size : %llu/%llu\n",
         (unsigned long long int)rgn->origin,
         (unsigned long long int)rgn->size);
}

template <> inline void print<>(pi_buff_rect_region rgn) {
  printf("pi_buff_rect_region width_bytes/height/depth : %llu/%llu/%llu\n",
         (unsigned long long int)rgn->width_bytes,
         (unsigned long long int)rgn->height_scalar,
         (unsigned long long int)rgn->depth_scalar);
}

template <> inline void print<>(pi_buff_rect_offset off) {
  printf("pi_buff_rect_offset x_bytes/y/z : %llu/%llu/%llu\n",
         (unsigned long long int)off->x_bytes,
         (unsigned long long int)off->y_scalar,
         (unsigned long long int)off->z_scalar);
}

template <> inline void print<>(pi_image_region rgn) {
  printf("pi_image_region width/height/depth : %llu/%llu/%llu\n",
         (unsigned long long int)rgn->width,
         (unsigned long long int)rgn->height,
         (unsigned long long int)rgn->depth);
}

template <> inline void print<>(pi_image_offset off) {
  printf("pi_image_offset x/y/z : %llu/%llu/%llu\n",
         (unsigned long long int)off->x, (unsigned long long int)off->y,
         (unsigned long long int)off->z);
}

template <> inline void print<>(const pi_image_desc *desc) {
  printf("image_desc w/h/d : %llu / %llu / %llu  --  arrSz/row/slice : %llu / "
         "%llu "
         "/ %llu"
         "  --  num_mip_lvls/num_smpls/image_type : %lld / %lld / %lld\n",
         (unsigned long long int)desc->image_width,
         (unsigned long long int)desc->image_height,
         (unsigned long long int)desc->image_depth,
         (unsigned long long int)desc->image_array_size,
         (unsigned long long int)desc->image_row_pitch,
         (unsigned long long int)desc->image_slice_pitch,
         (long long int)desc->num_mip_levels, (long long int)desc->num_samples,
         (long long int)desc->image_type);
}

template <> inline void print<>(PiResult val) {
  printf("pi_result : ");
  if (val == PI_SUCCESS)
    printf("PI_SUCCESS\n");
  else
    printf("%lld\n", (long long int)val);
}

// cout does not resolve a nullptr.
template <> inline void print<>(std::nullptr_t) { printf("<nullptr>\n"); }

template <> inline void print<>(char *val) {
  printf("<char * > : ");
  print_pointer(val, true);
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
  fflush(stdout);
}

template <typename T> struct printOut {
  printOut(T) {}
}; // Do nothing

template <> struct printOut<PiEvent *> {
  printOut(PiEvent *val) {
    printf("\t[out]pi_event * : ");
    print_pointer(val, false);
    if (val) {
      printf("[ ");
      print_pointer(*val, false);
      printf(" ... ]");
    } else {
      printf("[ nullptr ]");
    }
    printf("\n");
  }
};

template <> struct printOut<PiMem *> {
  printOut(PiMem *val) {
    printf("\t[out]pi_mem * : ");
    print_pointer(val, false);
    if (val) {
      printf("[ ");
      print_pointer(*val, false);
      printf(" ... ]");
    } else {
      printf("[ nullptr ]");
    }
    printf("\n");
  }
};

template <> struct printOut<void *> {
  printOut(void *val) {
    printf("\t[out]void * : ");
    print_pointer(val, true);
  }
};

template <typename T> struct printOut<T **> {
  printOut(T **val) {
    printf("\t[out]<unknown> ** : ");
    print_pointer(val, false);
    if (val) {
      printf("[ ");
      print_pointer(*val, false);
      printf(" ... ]");
    } else {
      printf("[ nullptr ]");
    }
    printf("\n");
  }
};

inline void printOuts(void) {}
template <typename Arg0, typename... Args>
void printOuts(Arg0 arg0, Args... args) {
  using T = decltype(arg0);
  printOut<T> a(arg0);
  printOuts(std::forward<Args>(args)...);
  fflush(stdout);
}

} // namespace pi
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
