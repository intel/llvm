//==--- fallback-cextmath.cpp - fallback implementation of math functions
//-----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device_math.h"

#ifdef __SPIR__

DEVICE_EXTERN_C
float __devicelib_erfinvf(float a) {
  float p, r, t;
  t = __spirv_ocl_fma(a, 0.0f - a, 1.0f);
  t = __spirv_ocl_log(t);
  if (__spirv_ocl_fabs(t) > 6.125f) {           // maximum ulp error = 2.35793
    p = 3.03697567e-10f;                        //  0x1.4deb44p-32
    p = __spirv_ocl_fma(p, t, 2.93243101e-8f);  //  0x1.f7c9aep-26
    p = __spirv_ocl_fma(p, t, 1.22150334e-6f);  //  0x1.47e512p-20
    p = __spirv_ocl_fma(p, t, 2.84108955e-5f);  //  0x1.dca7dep-16
    p = __spirv_ocl_fma(p, t, 3.93552968e-4f);  //  0x1.9cab92p-12
    p = __spirv_ocl_fma(p, t, 3.02698812e-3f);  //  0x1.8cc0dep-9
    p = __spirv_ocl_fma(p, t, 4.83185798e-3f);  //  0x1.3ca920p-8
    p = __spirv_ocl_fma(p, t, -2.64646143e-1f); // -0x1.0eff66p-2
    p = __spirv_ocl_fma(p, t, 8.40016484e-1f);  //  0x1.ae16a4p-1
  } else {                                      // maximum ulp error = 2.35002
    p = 5.43877832e-9f;                         //  0x1.75c000p-28
    p = __spirv_ocl_fma(p, t, 1.43285448e-7f);  //  0x1.33b402p-23
    p = __spirv_ocl_fma(p, t, 1.22774793e-6f);  //  0x1.499232p-20
    p = __spirv_ocl_fma(p, t, 1.12963626e-7f);  //  0x1.e52cd2p-24
    p = __spirv_ocl_fma(p, t, -5.61530760e-5f); // -0x1.d70bd0p-15
    p = __spirv_ocl_fma(p, t, -1.47697632e-4f); // -0x1.35be90p-13
    p = __spirv_ocl_fma(p, t, 2.31468678e-3f);  //  0x1.2f6400p-9
    p = __spirv_ocl_fma(p, t, 1.15392581e-2f);  //  0x1.7a1e50p-7
    p = __spirv_ocl_fma(p, t, -2.32015476e-1f); // -0x1.db2aeep-3
    p = __spirv_ocl_fma(p, t, 8.86226892e-1f);  //  0x1.c5bf88p-1
  }
  r = a * p;
  return r;
}

#endif // __SPIR__
