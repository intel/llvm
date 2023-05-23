//==--------------------------- imf_fp32_dl.cpp -  -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../device_imf.hpp"
#ifdef __LIBDEVICE_IMF_ENABLED__

DEVICE_EXTERN_C_INLINE int32_t __devicelib_imf_abs(int32_t x) {
  return (x >= 0) ? x : -x;
}

#endif /*__LIBDEVICE_IMF_ENABLED__*/
