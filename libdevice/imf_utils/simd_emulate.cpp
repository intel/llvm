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

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vaddus2(unsigned int x, unsigned int y) {
  uint16_t res_buf[2] = {
      0,
  };
  uint32_t tmp;
  uint16_t x_tmp, y_tmp;
  for (size_t idx = 0; idx < 2; ++idx) {
    x_tmp = __get_bytes_by_index<unsigned int, uint16_t>(x, idx);
    y_tmp = __get_bytes_by_index<unsigned int, uint16_t>(y, idx);
    tmp = x_tmp + y_tmp;
    if (tmp > 65535)
      res_buf[idx] = 0xFFFF;
    else
      res_buf[idx] = static_cast<uint16_t>(tmp);
  }
  return __assemble_integral_value<unsigned, uint16_t, 2>(res_buf);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vaddus4(unsigned int x, unsigned int y) {
  uint8_t res_buf[4] = {
      0,
  };
  uint16_t tmp;
  uint8_t x_tmp, y_tmp;
  for (size_t idx = 0; idx < 4; ++idx) {
    x_tmp = __get_bytes_by_index<unsigned int, uint8_t>(x, idx);
    y_tmp = __get_bytes_by_index<unsigned int, uint8_t>(y, idx);
    tmp = x_tmp + y_tmp;
    if (tmp > 255)
      res_buf[idx] = 0xFF;
    else
      res_buf[idx] = static_cast<uint8_t>(tmp);
  }
  return __assemble_integral_value<unsigned, uint8_t, 4>(res_buf);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vavgs2(unsigned int x, unsigned int y) {
  uint16_t res_buf[2] = {
      0,
  };
  int16_t x_tmp, y_tmp;
  for (size_t idx = 0; idx < 2; ++idx) {
    x_tmp = __bit_cast<int16_t>(
        __get_bytes_by_index<unsigned int, uint16_t>(x, idx));
    y_tmp = __bit_cast<int16_t>(
        __get_bytes_by_index<unsigned int, uint16_t>(y, idx));
    res_buf[idx] = __bit_cast<uint16_t>(__shadd(x_tmp, y_tmp));
  }
  return __assemble_integral_value<unsigned, uint16_t, 2>(res_buf);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vavgs4(unsigned int x, unsigned int y) {
  uint8_t res_buf[4] = {
      0,
  };
  int8_t x_tmp, y_tmp;
  for (size_t idx = 0; idx < 4; ++idx) {
    x_tmp =
        __bit_cast<int8_t>(__get_bytes_by_index<unsigned int, uint8_t>(x, idx));
    y_tmp =
        __bit_cast<int8_t>(__get_bytes_by_index<unsigned int, uint8_t>(y, idx));
    res_buf[idx] = __bit_cast<uint8_t>(__shadd(x_tmp, y_tmp));
  }
  return __assemble_integral_value<unsigned, uint8_t, 4>(res_buf);
}

template <typename Tp> class __min_op {
public:
  Tp operator()(const Tp &x, const Tp &y) { return (x < y) ? x : y; }
};

template <typename Tp> class __max_op {
public:
  Tp operator()(const Tp &x, const Tp &y) { return (x > y) ? x : y; }
};

template <typename Tp> class __eq_op {
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x, const Tp &y) {
    return ((x == y) ? static_cast<UTp>(-1) : 0);
  }
};

template <typename Tp> class __neq_op {
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x, const Tp &y) {
    return ((x != y) ? static_cast<UTp>(-1) : 0);
  }
};

template <typename Tp> class __set_eq_op {
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x, const Tp &y) { return ((x == y) ? 1 : 0); }
};

template <typename Tp> class __set_neq_op {
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x, const Tp &y) { return ((x != y) ? 1 : 0); }
};

template <typename Tp> class __gt_op {
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x, const Tp &y) {
    return ((x > y) ? static_cast<UTp>(-1) : 0);
  }
};

template <typename Tp> class __set_gt_op {
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x, const Tp &y) { return ((x > y) ? 1 : 0); }
};

template <typename Tp> class __ge_op {
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x, const Tp &y) {
    return ((x >= y) ? static_cast<UTp>(-1) : 0);
  }
};

template <typename Tp> class __set_ge_op {
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x, const Tp &y) { return ((x >= y) ? 1 : 0); }
};

template <typename Tp> class __lt_op {
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x, const Tp &y) {
    return ((x < y) ? static_cast<UTp>(-1) : 0);
  }
};

template <typename Tp> class __set_lt_op {
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x, const Tp &y) { return ((x < y) ? 1 : 0); }
};

template <typename Tp> class __le_op {
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x, const Tp &y) {
    return ((x <= y) ? static_cast<UTp>(-1) : 0);
  }
};

template <typename Tp> class __set_le_op {
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x, const Tp &y) { return ((x <= y) ? 1 : 0); }
};

template <typename Tp, size_t N, template <typename> class BinaryOp>
static inline unsigned int __internal_vs_binary_op(unsigned int x,
                                                   unsigned int y) {
  static_assert(std::is_same<Tp, int16_t>::value ||
                    std::is_same<Tp, int8_t>::value,
                "__internal_vs_binary_op only accept int8_t and int16_t.");
  static_assert(sizeof(Tp) * N == sizeof(unsigned int),
                "__internal_vs_binary_op size mismatch");
  typedef typename std::make_unsigned<Tp>::type UTp;
  UTp res_buf[N] = {
      0,
  };
  Tp x_tmp, y_tmp;
  BinaryOp<Tp> b_op;
  for (size_t idx = 0; idx < N; ++idx) {
    x_tmp = __bit_cast<Tp>(__get_bytes_by_index<unsigned int, UTp>(x, idx));
    y_tmp = __bit_cast<Tp>(__get_bytes_by_index<unsigned int, UTp>(y, idx));
    res_buf[idx] = b_op(x_tmp, y_tmp);
  }
  return __assemble_integral_value<unsigned, UTp, N>(res_buf);
}

template <typename Tp, size_t N, template <typename> class BinaryOp>
static inline unsigned int __internal_vu_binary_op(unsigned int x,
                                                   unsigned int y) {
  static_assert(std::is_same<Tp, uint16_t>::value ||
                    std::is_same<Tp, uint8_t>::value,
                "__internal_vu_binary_op only accept uint8_t and uint16_t.");
  static_assert(sizeof(Tp) * N == sizeof(unsigned int),
                "__internal_vu_binary_op size mismatch");
  Tp res_buf[N] = {
      0,
  };
  Tp x_tmp, y_tmp;
  BinaryOp<Tp> b_op;
  for (size_t idx = 0; idx < N; ++idx) {
    x_tmp = __get_bytes_by_index<unsigned int, Tp>(x, idx);
    y_tmp = __get_bytes_by_index<unsigned int, Tp>(y, idx);
    res_buf[idx] = b_op(x_tmp, y_tmp);
  }
  return __assemble_integral_value<unsigned, Tp, N>(res_buf);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpeq2(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint16_t, 2, __eq_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpeq4(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint8_t, 4, __eq_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpges2(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int16_t, 2, __ge_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpges4(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int8_t, 4, __ge_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpgeu2(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint16_t, 2, __ge_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpgeu4(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint8_t, 4, __ge_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpgts2(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int16_t, 2, __gt_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpgts4(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int8_t, 4, __gt_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpgtu2(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint16_t, 2, __gt_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpgtu4(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint8_t, 4, __gt_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmples2(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int16_t, 2, __le_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmples4(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int8_t, 4, __le_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpleu2(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint16_t, 2, __le_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpleu4(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint8_t, 4, __le_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmplts2(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int16_t, 2, __lt_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmplts4(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int8_t, 4, __le_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpltu2(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint16_t, 2, __lt_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpltu4(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint8_t, 4, __lt_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpne2(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint16_t, 2, __neq_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpne4(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint8_t, 4, __neq_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vmaxs2(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int16_t, 2, __max_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vmaxs4(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int8_t, 4, __max_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vmaxu2(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint16_t, 2, __max_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vmaxu4(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint8_t, 4, __max_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vmins2(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int16_t, 2, __min_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vmins4(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int8_t, 4, __min_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vminu2(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint16_t, 2, __min_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vminu4(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint8_t, 4, __min_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vseteq2(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint16_t, 2, __set_eq_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vseteq4(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint8_t, 4, __set_eq_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetne2(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint16_t, 2, __set_neq_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetne4(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint8_t, 4, __set_neq_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetges2(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int16_t, 2, __set_ge_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetges4(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int8_t, 4, __set_ge_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetgeu2(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint16_t, 2, __set_ge_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetgeu4(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint8_t, 4, __set_ge_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetgts2(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int16_t, 2, __set_gt_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetgts4(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int8_t, 4, __set_gt_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetgtu2(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint16_t, 2, __set_gt_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetgtu4(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint8_t, 4, __set_gt_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetles2(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int16_t, 2, __set_le_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetles4(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int8_t, 4, __set_le_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetleu2(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint16_t, 2, __set_le_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetleu4(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint8_t, 4, __set_le_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetlts2(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int16_t, 2, __set_lt_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetlts4(unsigned int x, unsigned int y) {
  return __internal_vs_binary_op<int8_t, 4, __set_lt_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetltu2(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint16_t, 2, __set_lt_op>(x, y);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetltu4(unsigned int x, unsigned int y) {
  return __internal_vu_binary_op<uint8_t, 4, __set_lt_op>(x, y);
}
#endif
