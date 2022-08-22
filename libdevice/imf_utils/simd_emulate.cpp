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

template <typename Tp> struct __twice_size;
template <typename Tp> using __twice_size_t = typename __twice_size<Tp>::type;
template <typename Tp> struct __twice_size_tag {
  using type = Tp;
};

template <> struct __twice_size<int8_t> : __twice_size_tag<int16_t> {};
template <> struct __twice_size<int16_t> : __twice_size_tag<int32_t> {};
template <> struct __twice_size<uint8_t> : __twice_size_tag<uint16_t> {};
template <> struct __twice_size<uint16_t> : __twice_size_tag<uint32_t> {};

template <typename Tp> class __abs_op {
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x) { return __abs(x); }
};

template <typename Tp> class __abss_op {
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x) {
    if (x == std::numeric_limits<Tp>::min())
      return std::numeric_limits<Tp>::max();
    else
      return __abs(x);
  }
};

template <typename Tp> class __neg_op {
  static_assert(std::is_same<int8_t, Tp>::value ||
                    std::is_same<int16_t, Tp>::value,
                "Tp can only accept int8_t, int16_t for __neg_op");
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x) { return static_cast<UTp>(-x); }
};

template <typename Tp> class __negss_op {
  static_assert(std::is_same<int8_t, Tp>::value ||
                    std::is_same<int16_t, Tp>::value,
                "Tp can only accept int8_t, int16_t for __negss_op");
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x) {
    UTp tx = static_cast<UTp>(-x);
    if (x == std::numeric_limits<Tp>::min())
      tx = static_cast<UTp>(std::numeric_limits<Tp>::max());
    return tx;
  }
};

template <typename Tp, size_t N, template <typename> class UnaryOp>
static inline unsigned int __internal_v_unary_op(unsigned int x) {
  static_assert(std::is_integral<Tp>::value &&
                    (sizeof(Tp) == 1 || sizeof(Tp) == 2),
                "__internal_v_unary_op accepts 1/2 byte integer type.");
  static_assert(sizeof(Tp) * N == sizeof(unsigned int),
                "__internal_v_unary_op size mismatch");
  typedef typename std::make_unsigned<Tp>::type UTp;
  UTp res_buf[N] = {
      0,
  };
  Tp x_tmp;
  UnaryOp<Tp> u_op;
  for (size_t idx = 0; idx < N; ++idx) {
    x_tmp = static_cast<Tp>(__get_bytes_by_index<unsigned int, UTp>(x, idx));
    res_buf[idx] = u_op(x_tmp);
  }
  return __assemble_integral_value<unsigned, UTp, N>(res_buf);
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

template <typename Tp> class __abs_diff_s_op {
  static_assert(std::is_same<int8_t, Tp>::value ||
                    std::is_same<int16_t, Tp>::value,
                "Tp can only accept int8_t, int16_t for __abs_diff_s_op");
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x, const Tp &y) {
    __twice_size_t<Tp> tx = x, ty = y;
    tx -= ty;
    return static_cast<UTp>(__abs(tx));
  }
};

template <typename Tp> class __abs_diff_u_op {
  static_assert(std::is_same<uint8_t, Tp>::value ||
                    std::is_same<uint16_t, Tp>::value,
                "Tp can only accept uint8_t, uint16_t for __abs_diff_u_op");

public:
  Tp operator()(Tp &x, Tp &y) {
    if (x < y)
      __swap(x, y);
    x -= y;
    return x;
  }
};

template <typename Tp> class __add_op {
public:
  Tp operator()(const Tp &x, const Tp &y) { return x + y; }
};

template <typename Tp> class __add_us_op {
  static_assert(std::is_same<uint8_t, Tp>::value ||
                    std::is_same<uint16_t, Tp>::value,
                "Tp can only accept uint8_t, uint16_t for __add_us_op");

public:
  Tp operator()(const Tp &x, const Tp &y) {
    __twice_size_t<Tp> z = x + y;
    if (z > std::numeric_limits<Tp>::max())
      return std::numeric_limits<Tp>::max();
    else
      return static_cast<Tp>(z);
  }
};

// Clang will optimize this function with llvm.sadd.sat intrinsic which
// can't be handled by llvm-spirv translator, so using turn off clang
// optimization for this function to avoid llvm-spirv crash.
#pragma clang optimize off
template <typename Tp> class __add_ss_op {
  static_assert(std::is_same<int8_t, Tp>::value ||
                    std::is_same<int16_t, Tp>::value,
                "Tp can only accept int8_t, int16_t for __add_ss_op");
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x, const Tp &y) {
    __twice_size_t<Tp> z = x + y;
    __max_op<__twice_size_t<Tp>> __max_val;
    __min_op<__twice_size_t<Tp>> __min_val;
    return static_cast<UTp>(
        __min_val(__max_val(z, std::numeric_limits<Tp>::min()),
                  std::numeric_limits<Tp>::max()));
  }
};
#pragma clang optimize on

template <typename Tp> class __sub_op {
public:
  Tp operator()(const Tp &x, const Tp &y) { return x - y; }
};

template <typename Tp> class __sub_us_op {
  static_assert(std::is_same<uint8_t, Tp>::value ||
                    std::is_same<uint16_t, Tp>::value,
                "Tp can only accept uint8_t, uint16_t for __add_us_op");

public:
  Tp operator()(const Tp &x, const Tp &y) {
    if (x < y)
      return 0;
    else
      return x - y;
  }
};

// Clang will optimize this function with llvm.sadd.sat intrinsic which
// can't be handled by llvm-spirv translator, so using turn off clang
// optimization for this function to avoid llvm-spirv crash.
#pragma clang optimize off
template <typename Tp> class __sub_ss_op {
  static_assert(std::is_same<int8_t, Tp>::value ||
                    std::is_same<int16_t, Tp>::value,
                "Tp can only accept int8_t, int16_t for __add_ss_op");
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x, const Tp &y) {
    __twice_size_t<Tp> z = x - y;
    __max_op<__twice_size_t<Tp>> __max_val;
    __min_op<__twice_size_t<Tp>> __min_val;
    return static_cast<UTp>(
        __min_val(__max_val(z, std::numeric_limits<Tp>::min()),
                  std::numeric_limits<Tp>::max()));
  }
};
#pragma clang optimize on

template <typename Tp> class __avgs_op {
  static_assert(std::is_same<int8_t, Tp>::value ||
                    std::is_same<int16_t, Tp>::value,
                "Tp can only accept int8_t, int16_t for __avgs_op");
  typedef typename std::make_unsigned<Tp>::type UTp;

public:
  UTp operator()(const Tp &x, const Tp &y) {
    return static_cast<UTp>(__srhadd(x, y));
  }
};

template <typename Tp> class __avgu_op {
  static_assert(std::is_same<uint8_t, Tp>::value ||
                    std::is_same<uint16_t, Tp>::value,
                "Tp can only accept uint8_t, uint16_t for __avgu_op");

public:
  Tp operator()(const Tp &x, const Tp &y) { return __urhadd(x, y); }
};

template <typename Tp> class __uhadd_op {
  static_assert(std::is_same<uint8_t, Tp>::value ||
                    std::is_same<uint16_t, Tp>::value,
                "Tp can only accept uint8_t, uint16_t for __uhadd_op");

public:
  Tp operator()(const Tp &x, const Tp &y) { return __uhadd(x, y); }
};

template <typename Tp, size_t N, template <typename> class BinaryOp>
static inline unsigned int __internal_v_binary_op(unsigned int x,
                                                  unsigned int y) {
  static_assert(std::is_integral<Tp>::value &&
                    (sizeof(Tp) == 1 || sizeof(Tp) == 2),
                "__internal_v_binary_op accepts 1/2 byte integer type.");
  static_assert(sizeof(Tp) * N == sizeof(unsigned int),
                "__internal_v_binary_op size mismatch");
  typedef typename std::make_unsigned<Tp>::type UTp;
  UTp res_buf[N] = {
      0,
  };
  Tp x_tmp, y_tmp;
  BinaryOp<Tp> b_op;
  for (size_t idx = 0; idx < N; ++idx) {
    x_tmp = static_cast<Tp>(__get_bytes_by_index<unsigned int, UTp>(x, idx));
    y_tmp = static_cast<Tp>(__get_bytes_by_index<unsigned int, UTp>(y, idx));
    res_buf[idx] = b_op(x_tmp, y_tmp);
  }
  return __assemble_integral_value<unsigned, UTp, N>(res_buf);
}

// Split 32-bit into 2 parts, each consisting of 16 bits, compute absolute
// value for each part and assemble the results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vabs2(unsigned int x) {
  return __internal_v_unary_op<int16_t, 2, __abs_op>(x);
}

// Split 32-bit into 4 parts, each consisting of 8 bits, compute absolute
// value for each part and assemble the results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vabs4(unsigned int x) {
  return __internal_v_unary_op<int8_t, 4, __abs_op>(x);
}

// Split 32-bit into 2 parts, each consisting of 16 bits, compute absolute
// value with signed saturation for each part and assemble the results
// into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vabsss2(unsigned int x) {
  return __internal_v_unary_op<int16_t, 2, __abss_op>(x);
}

// Split 32-bit into 4 parts, each consisting of 8 bits, compute absolute
// value with signed saturation for each part and assemble the results
// into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vabsss4(unsigned int x) {
  return __internal_v_unary_op<int8_t, 4, __abss_op>(x);
}

// Split 32-bit into 2 parts, each consisting of 16 bits, compute negative
// value for each part and assemble the results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vneg2(unsigned int x) {
  return __internal_v_unary_op<int16_t, 2, __neg_op>(x);
}

// Split 32-bit into 4 parts, each consisting of 8 bits, compute negative
// value for each part and assemble the results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vneg4(unsigned int x) {
  return __internal_v_unary_op<int8_t, 4, __neg_op>(x);
}

// Split 32-bit into 2 parts, each consisting of 16 bits, compute negative
// value with signed saturation for each part and assemble the results into
// 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vnegss2(unsigned int x) {
  return __internal_v_unary_op<int16_t, 2, __negss_op>(x);
}

// Split 32-bit into 4 parts, each consisting of 8 bits, compute negative
// value with signed saturation for each part and assemble the results into
// 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vnegss4(unsigned int x) {
  return __internal_v_unary_op<int8_t, 4, __negss_op>(x);
}

// Split 32-bit into 2 parts, each part is sigend 16-bit int, compute absolute
// difference for corresponding parts and assemble the results into 32-bit
// unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vabsdiffs2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int16_t, 2, __abs_diff_s_op>(x, y);
}

// Split 32-bit into 4 parts, each part is sigend 8-bit int, compute absolute
// difference for corresponding parts and assemble the results into 32-bit
// unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vabsdiffs4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int8_t, 4, __abs_diff_s_op>(x, y);
}

// Split 32-bit into 2 parts, each part is unsigend 16-bit int, compute
// absolute difference for corresponding parts and assemble the results
// into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vabsdiffu2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __abs_diff_u_op>(x, y);
}

// Split 32-bit into 4 parts, each part is unsigend 8-bit int, compute
// absolute difference for corresponding parts and assemble the results
// into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vabsdiffu4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __abs_diff_u_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits, compute
// unsigned addition for corresponding parts and assemble the results
// into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vadd2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __add_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits, compute
// unsigned addition for corresponding parts and assemble the results
// into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vadd4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __add_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits, compute
// addition with signed saturation for corresponding parts and assemble
// the results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vaddss2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int16_t, 2, __add_ss_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits, compute
// addition with signed saturation for corresponding parts and assemble
// the results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vaddss4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int8_t, 4, __add_ss_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits, compute
// addition with unsigned saturation for corresponding parts and assemble
// the results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vaddus2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __add_us_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits, compute
// addition with unsigned saturation for corresponding parts and assemble
// the results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vaddus4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __add_us_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits, compute
// subtraction with wrap-round for corresponding parts and assemble
// the results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsub2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __sub_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits, compute
// subtraction with wrap-round for corresponding parts and assemble
// the results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsub4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __sub_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits, compute
// subtraction with signed saturation for corresponding parts and assemble
// the results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsubss2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int16_t, 2, __sub_ss_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits, compute
// subtraction with signed saturation for corresponding parts and assemble
// the results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsubss4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int8_t, 4, __sub_ss_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits, compute
// subtraction with unsigned saturation for corresponding parts and assemble
// the results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsubus2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __sub_us_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits, compute
// subtraction with unsigned saturation for corresponding parts and assemble
// the results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsubus4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __sub_us_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits, compute
// signed rounded average for corresponding parts and assemble the results
// into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vavgs2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int16_t, 2, __avgs_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits, compute
// signed rounded average for corresponding parts and assemble the results
// into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vavgs4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int8_t, 4, __avgs_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits, compute
// unsigned rounded average for corresponding parts and assemble the results
// into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vavgu2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __avgu_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits, compute
// unsigned rounded average for corresponding parts and assemble the results
// into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vavgu4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __avgu_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits, compute
// unsigned average for corresponding parts and assemble the results
// into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vhaddu2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __uhadd_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits, compute
// unsigned average for corresponding parts and assemble the results
// into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vhaddu4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __uhadd_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits. Compare
// corresponding parts, return 0xFFFF if they are equal, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpeq2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __eq_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits. Compare
// corresponding parts, return 0xFF if they are equal, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpeq4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __eq_op>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit signed int. For corresponding
// part from x and y, return 0xFFFF if x >= y, otherwise return 0. Assemble
// partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpges2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int16_t, 2, __ge_op>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit signed int. For corresponding
// part from x and y, return 0xFF if x >= y, otherwise return 0. Assemble
// partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpges4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int8_t, 4, __ge_op>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit unsigned int. For
// corresponding part from x and y, return 0xFFFF if x >= y, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpgeu2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __ge_op>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit unsigned int. For
// corresponding part from x and y, return 0xFF if x >= y, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpgeu4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __ge_op>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit signed int. For corresponding
// part from x and y, return 0xFFFF if x > y, otherwise return 0. Assemble
// partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpgts2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int16_t, 2, __gt_op>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit signed int. For corresponding
// part from x and y, return 0xFF if x > y, otherwise return 0. Assemble
// partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpgts4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int8_t, 4, __gt_op>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit unsigned int. For
// corresponding part from x and y, return 0xFFFF if x > y, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpgtu2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __gt_op>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit unsigned int. For
// corresponding part from x and y, return 0xFF if x > y, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpgtu4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __gt_op>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit signed int. For corresponding
// part from x and y, return 0xFFFF if x <= y, otherwise return 0. Assemble
// partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmples2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int16_t, 2, __le_op>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit signed int. For corresponding
// part from x and y, return 0xFF if x <= y, otherwise return 0. Assemble
// partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmples4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int8_t, 4, __le_op>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit unsigned int. For
// corresponding part from x and y, return 0xFFFF if x <= y, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpleu2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __le_op>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit unsigned int. For
// corresponding part from x and y, return 0xFF if x <= y, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpleu4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __le_op>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit signed int. For corresponding
// part from x and y, return 0xFFFF if x < y, otherwise return 0. Assemble
// partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmplts2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int16_t, 2, __lt_op>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit signed int. For corresponding
// part from x and y, return 0xFF if x < y, otherwise return 0. Assemble
// partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmplts4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int8_t, 4, __lt_op>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit unsigned int. For
// corresponding part from x and y, return 0xFFFF if x < y, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpltu2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __lt_op>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit unsigned int. For
// corresponding part from x and y, return 0xFF if x < y, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpltu4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __lt_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits. Compare
// corresponding parts, return 0xFFFF if they are not equal, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpne2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __neq_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits. Compare
// corresponding parts, return 0xFF if they are not equal, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vcmpne4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __neq_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits.
// For corresponding parts, compute signed maximum value and assemble partial
// results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vmaxs2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int16_t, 2, __max_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits. For corresponding
// parts, compute signed maximum value and assemble partial results into 32-bit
// unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vmaxs4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int8_t, 4, __max_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits.
// For corresponding parts, compute unsigned maximum value and assemble partial
// results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vmaxu2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __max_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits. For corresponding
// parts, compute unsigned maximum value and assemble partial results into
// 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vmaxu4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __max_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits.
// For corresponding parts, compute signed minimum value and assemble partial
// results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vmins2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int16_t, 2, __min_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits. For corresponding
// parts, compute signed minimum value and assemble partial results into 32-bit
// unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vmins4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int8_t, 4, __min_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits.
// For corresponding parts, compute unsigned minimum value and assemble partial
// results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vminu2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __min_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits. For corresponding
// parts, compute unsigned minimum value and assemble partial results into
// 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vminu4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __min_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits. Compare
// corresponding parts, return 1 if they are equal, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vseteq2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __set_eq_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits. Compare
// corresponding parts, return 1 if they are equal, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vseteq4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __set_eq_op>(x, y);
}

// Split 32-bit into 2 parts, each part consisting of 16 bits. Compare
// corresponding parts, return 1 if they are not equal, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetne2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __set_neq_op>(x, y);
}

// Split 32-bit into 4 parts, each part consisting of 8 bits. Compare
// corresponding parts, return 1 if they are not equal, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetne4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __set_neq_op>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit signed int. For corresponding
// part from x and y, return 1 if x >= y, otherwise return 0. Assemble
// partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetges2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int16_t, 2, __set_ge_op>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit signed int. For corresponding
// part from x and y, return 1 if x >= y, otherwise return 0. Assemble
// partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetges4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int8_t, 4, __set_ge_op>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit unsigned int. For
// corresponding part from x and y, return 1 if x >= y, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetgeu2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __set_ge_op>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit unsigned int. For
// corresponding part from x and y, return 1 if x >= y, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetgeu4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __set_ge_op>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit signed int. For corresponding
// part from x and y, return 1 if x > y, otherwise return 0. Assemble
// partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetgts2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int16_t, 2, __set_gt_op>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit signed int. For corresponding
// part from x and y, return 1 if x > y, otherwise return 0. Assemble
// partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetgts4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int8_t, 4, __set_gt_op>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit unsigned int. For
// corresponding part from x and y, return 1 if x > y, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetgtu2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __set_gt_op>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit unsigned int. For
// corresponding part from x and y, return 1 if x > y, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetgtu4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __set_gt_op>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit signed int. For corresponding
// part from x and y, return 1 if x <= y, otherwise return 0. Assemble
// partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetles2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int16_t, 2, __set_le_op>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit signed int. For corresponding
// part from x and y, return 1 if x <= y, otherwise return 0. Assemble
// partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetles4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int8_t, 4, __set_le_op>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit unsigned int. For
// corresponding part from x and y, return 1 if x <= y, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetleu2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __set_le_op>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit unsigned int. For
// corresponding part from x and y, return 1 if x <= y, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetleu4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __set_le_op>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit signed int. For corresponding
// part from x and y, return 1 if x < y, otherwise return 0. Assemble
// partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetlts2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int16_t, 2, __set_lt_op>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit signed int. For corresponding
// part from x and y, return 1 if x < y, otherwise return 0. Assemble
// partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetlts4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<int8_t, 4, __set_lt_op>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit unsigned int. For
// corresponding part from x and y, return 1 if x < y, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetltu2(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint16_t, 2, __set_lt_op>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit unsigned int. For
// corresponding part from x and y, return 1 if x < y, otherwise return 0.
// Assemble partial results into 32-bit unsigned int.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsetltu4(unsigned int x, unsigned int y) {
  return __internal_v_binary_op<uint8_t, 4, __set_lt_op>(x, y);
}

template <typename Tp, size_t N>
static inline unsigned int __internal_v_sad_op(unsigned int x, unsigned int y) {
  static_assert(std::is_integral<Tp>::value &&
                    (sizeof(Tp) == 1 || sizeof(Tp) == 2),
                "__internal_v_sad_op accepts 1/2 byte integer type.");
  static_assert(sizeof(Tp) * N == sizeof(unsigned int),
                "__internal_v_sad_op size mismatch");
  typedef typename std::make_unsigned<Tp>::type UTp;
  unsigned int res = 0;
  typedef __twice_size_t<Tp> __TwiceTp;
  Tp x_tmp, y_tmp;
  for (size_t idx = 0; idx < N; ++idx) {
    x_tmp = static_cast<Tp>(__get_bytes_by_index<unsigned int, UTp>(x, idx));
    y_tmp = static_cast<Tp>(__get_bytes_by_index<unsigned int, UTp>(y, idx));
    if (x_tmp < y_tmp)
      __swap(x_tmp, y_tmp);
    res += static_cast<unsigned int>(static_cast<__TwiceTp>(x_tmp - y_tmp));
  }
  return res;
}

// Split 32-bit into 2 parts, each part is 16-bit signed int. For corresponding
// parts, compute absolute difference and sum them up.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsads2(unsigned int x, unsigned int y) {
  return __internal_v_sad_op<int16_t, 2>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit signed int. For corresponding
// parts, compute absolute difference and sum them up.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsads4(unsigned int x, unsigned int y) {
  return __internal_v_sad_op<int8_t, 4>(x, y);
}

// Split 32-bit into 2 parts, each part is 16-bit unsigned int. For
// corresponding parts, compute absolute difference and sum them up.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsadu2(unsigned int x, unsigned int y) {
  return __internal_v_sad_op<uint16_t, 2>(x, y);
}

// Split 32-bit into 4 parts, each part is 8-bit unsigned int. For
// corresponding parts, compute absolute difference and sum them up.
DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_vsadu4(unsigned int x, unsigned int y) {
  return __internal_v_sad_op<uint8_t, 4>(x, y);
}
#endif
