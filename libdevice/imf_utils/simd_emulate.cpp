//==------ simd_emulate.cpp - serial implementation to emulate simd functions
// ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../device_imf.hpp"
#ifdef __LIBDEVICE_IMF_ENABLED__

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vabs2(unsigned int x) {
  uint16_t res_buf[2] = {
      0,
  };
  for (size_t idx = 0; idx < 2; ++idx) {
    int16_t tmp = __bit_cast<int16_t>(
        __get_bytes_by_index<unsigned int, uint16_t>(x, idx));
    res_buf[idx] = __bit_cast<uint16_t>(__abs(tmp));
  }
  return __assemble_integral_value<unsigned int, uint16_t, 2>(res_buf);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vabs4(unsigned int x) {
  uint8_t res_buf[4] = {
      0,
  };
  for (size_t idx = 0; idx < 4; ++idx) {
    int8_t tmp =
        __bit_cast<int8_t>(__get_bytes_by_index<unsigned int, uint8_t>(x, idx));
    res_buf[idx] = __bit_cast<uint8_t>(__abs(tmp));
  }
  return __assemble_integral_value<unsigned int, uint8_t, 4>(res_buf);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vabsss2(unsigned int x) {
  uint16_t res_buf[2] = {
      0,
  };
  for (size_t idx = 0; idx < 2; ++idx) {
    uint16_t tmp = __get_bytes_by_index<unsigned int, uint16_t>(x, idx);
    if (tmp == 0x8000)
      res_buf[idx] = 0x7FFF;
    else {
      int16_t s_tmp = __bit_cast<int16_t>(tmp);
      res_buf[idx] = __bit_cast<uint16_t>(__abs(s_tmp));
    }
  }
  return __assemble_integral_value<unsigned int, uint16_t, 2>(res_buf);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vabsss4(unsigned int x) {
  uint8_t res_buf[4] = {
      0,
  };
  for (size_t idx = 0; idx < 4; ++idx) {
    uint8_t tmp = __get_bytes_by_index<unsigned int, uint8_t>(x, idx);
    if (tmp == 0x80)
      res_buf[idx] = 0x7F;
    else {
      int8_t s_tmp = __bit_cast<int8_t>(tmp);
      res_buf[idx] = __bit_cast<uint8_t>(__abs(s_tmp));
    }
  }
  return __assemble_integral_value<unsigned int, uint8_t, 4>(res_buf);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vabsdiffs2(unsigned int x, unsigned int y) {
  uint16_t res_buf[2] = {
      0,
  };
  int32_t x_tmp, y_tmp;
  for (size_t idx = 0; idx < 2; ++idx) {
    x_tmp = static_cast<int32_t>(__bit_cast<int16_t>(
        __get_bytes_by_index<unsigned int, uint16_t>(x, idx)));
    y_tmp = static_cast<int32_t>(__bit_cast<int16_t>(
        __get_bytes_by_index<unsigned int, uint16_t>(y, idx)));
    x_tmp -= y_tmp;
    res_buf[idx] = static_cast<uint16_t>(__abs(x_tmp));
  }
  return __assemble_integral_value<unsigned int, uint16_t, 2>(res_buf);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vabsdiffs4(unsigned int x, unsigned int y) {
  uint8_t res_buf[4] = {
      0,
  };
  int16_t x_tmp, y_tmp;
  for (size_t idx = 0; idx < 4; ++idx) {
    x_tmp = static_cast<int16_t>(__bit_cast<int8_t>(
        __get_bytes_by_index<unsigned int, uint8_t>(x, idx)));
    y_tmp = static_cast<int16_t>(__bit_cast<int8_t>(
        __get_bytes_by_index<unsigned int, uint8_t>(y, idx)));
    x_tmp -= y_tmp;
    res_buf[idx] = static_cast<uint8_t>(__abs(x_tmp));
  }
  return __assemble_integral_value<unsigned int, uint8_t, 4>(res_buf);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vabsdiffu2(unsigned int x, unsigned int y) {
  uint16_t res_buf[2] = {
      0,
  };
  uint16_t x_tmp, y_tmp;
  for (size_t idx = 0; idx < 2; ++idx) {
    x_tmp = __get_bytes_by_index<unsigned int, uint16_t>(x, idx);
    y_tmp = __get_bytes_by_index<unsigned int, uint16_t>(y, idx);
    if (x_tmp < y_tmp)
      __swap(x_tmp, y_tmp);
    x_tmp -= y_tmp;
    res_buf[idx] = x_tmp;
  }
  return __assemble_integral_value<unsigned int, uint16_t, 2>(res_buf);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vabsdiffu4(unsigned int x, unsigned int y) {
  uint8_t res_buf[4] = {
      0,
  };
  uint8_t x_tmp, y_tmp;
  for (size_t idx = 0; idx < 4; ++idx) {
    x_tmp = __get_bytes_by_index<unsigned int, uint8_t>(x, idx);
    y_tmp = __get_bytes_by_index<unsigned int, uint8_t>(y, idx);
    if (x_tmp < y_tmp)
      __swap(x_tmp, y_tmp);
    x_tmp -= y_tmp;
    res_buf[idx] = x_tmp;
  }
  return __assemble_integral_value<unsigned int, uint8_t, 4>(res_buf);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vadd2(unsigned int x, unsigned int y) {
  uint16_t res_buf[2] = {
      0,
  };

  uint32_t tmp;
  uint16_t x_tmp, y_tmp;
  for (size_t idx = 0; idx < 2; ++idx) {
    x_tmp = __get_bytes_by_index<unsigned int, uint16_t>(x, idx);
    y_tmp = __get_bytes_by_index<unsigned int, uint16_t>(y, idx);
    tmp = x_tmp + y_tmp;
    res_buf[idx] = __get_bytes_by_index<uint32_t, uint16_t>(tmp, 0);
  }
  return __assemble_integral_value<unsigned int, uint16_t, 2>(res_buf);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vadd4(unsigned int x, unsigned int y) {
  uint8_t res_buf[4] = {
      0,
  };

  uint16_t tmp;
  uint8_t x_tmp, y_tmp;
  for (size_t idx = 0; idx < 4; ++idx) {
    x_tmp = __get_bytes_by_index<unsigned int, uint8_t>(x, idx);
    y_tmp = __get_bytes_by_index<unsigned int, uint8_t>(y, idx);
    tmp = x_tmp + y_tmp;
    res_buf[idx] = __get_bytes_by_index<uint16_t, uint8_t>(tmp, 0);
  }
  return __assemble_integral_value<unsigned int, uint8_t, 4>(res_buf);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vaddss2(unsigned int x, unsigned int y) {
  uint16_t res_buf[2] = {
      0,
  };

  int32_t tmp;
  int16_t x_tmp, y_tmp;
  for (size_t idx = 0; idx < 2; ++idx) {
    x_tmp = __bit_cast<int16_t>(
        __get_bytes_by_index<unsigned int, uint16_t>(x, idx));
    y_tmp = __bit_cast<int16_t>(
        __get_bytes_by_index<unsigned int, uint16_t>(y, idx));
    tmp = x_tmp + y_tmp;
    if (tmp > 32767)
      res_buf[idx] = 0x7FFF;
    else if (tmp < -32768)
      res_buf[idx] = 0x8000;
    else
      res_buf[idx] = __get_bytes_by_index<uint32_t, uint16_t>(tmp, 0);
  }
  return __assemble_integral_value<unsigned int, uint16_t, 2>(res_buf);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vaddss4(unsigned int x, unsigned int y) {
  uint8_t res_buf[4] = {
      0,
  };

  int16_t tmp;
  int8_t x_tmp, y_tmp;
  for (size_t idx = 0; idx < 4; ++idx) {
    x_tmp =
        __bit_cast<int8_t>(__get_bytes_by_index<unsigned int, uint8_t>(x, idx));
    y_tmp =
        __bit_cast<int8_t>(__get_bytes_by_index<unsigned int, uint8_t>(y, idx));
    tmp = x_tmp + y_tmp;
    if (tmp > 127)
      res_buf[idx] = 0x7F;
    else if (tmp < -128)
      res_buf[idx] = 0x80;
    else
      res_buf[idx] = __get_bytes_by_index<uint16_t, uint8_t>(tmp, 0);
  }
  return __assemble_integral_value<unsigned int, uint8_t, 4>(res_buf);
}
#endif
