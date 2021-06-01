//==-------------------------- host_util.hpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions and definitions for implementing ESIMD intrinsics on host
//===----------------------------------------------------------------------===//

#pragma once

#ifndef __SYCL_DEVICE_ONLY__

#include <limits>

#define SIMDCF_ELEMENT_SKIP(i)

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
namespace half_impl {
class half;
} // namespace half_impl
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {
namespace emu {
namespace detail {

using half = sycl::detail::half_impl::half;
constexpr int sat_is_on = 1;

static long long abs(long long a) {
  if (a < 0) {
    return -a;
  } else {
    return a;
  }
}

template <typename RT> struct satur {
  template <typename T> static RT saturate(const T val, const int flags) {
    if ((flags & sat_is_on) == 0) {
      return (RT)val;
    }

    // min/max can be macros on Windows, so wrap them into parens to avoid their
    // expansion
    const RT t_max = (std::numeric_limits<RT>::max)();
    const RT t_min = (std::numeric_limits<RT>::min)();

    if (val > t_max) {
      return t_max;
    } else if ((val >= 0) && (t_min < 0)) {
      // RT is "signed" if t_min < 0
      // when comparing a signed and a unsigned variable, the signed one cast to
      // unsigned first.
      return (RT)val;
    } else if (val < t_min) {
      return t_min;
    } else {
      return (RT)val;
    }
  }
};

template <> struct satur<float> {
  template <typename T> static float saturate(const T val, const int flags) {
    if ((flags & sat_is_on) == 0) {
      return (float)val;
    }

    if (val < 0.) {
      return 0;
    } else if (val > 1.) {
      return 1.;
    } else {
      return (float)val;
    }
  }
};

template <> struct satur<double> {
  template <typename T> static double saturate(const T val, const int flags) {
    if ((flags & sat_is_on) == 0) {
      return (double)val;
    }

    if (val < 0.) {
      return 0;
    } else if (val > 1.) {
      return 1.;
    } else {
      return (double)val;
    }
  }
};

template <typename T1, bool B> struct SetSatur {
  static unsigned int set() { return 0; }
};

template <> struct SetSatur<float, true> {
  static unsigned int set() { return sat_is_on; }
};

template <> struct SetSatur<double, true> {
  static unsigned int set() { return sat_is_on; }
};

// used for intermediate type in dp4a emulation
template <typename T1, typename T2> struct restype_ex {
private:
  restype_ex();
};

template <> struct restype_ex<char, char> { using type = int; };
template <> struct restype_ex<char, unsigned char> { using type = int; };
template <> struct restype_ex<char, short> { using type = int; };
template <> struct restype_ex<char, unsigned short> { using type = int; };
template <> struct restype_ex<char, int> { using type = long long; };
template <> struct restype_ex<char, unsigned int> { using type = long long; };
template <> struct restype_ex<char, half> { using type = half; };
template <> struct restype_ex<char, float> { using type = float; };
template <> struct restype_ex<char, double> { using type = double; };

template <> struct restype_ex<unsigned char, char> { using type = int; };
template <> struct restype_ex<unsigned char, unsigned char> {
  using type = int;
};
template <> struct restype_ex<unsigned char, short> { using type = int; };
template <> struct restype_ex<unsigned char, unsigned short> {
  using type = int;
};
template <> struct restype_ex<unsigned char, int> { using type = long long; };
template <> struct restype_ex<unsigned char, unsigned int> {
  using type = long long;
};
template <> struct restype_ex<unsigned char, half> { using type = half; };
template <> struct restype_ex<unsigned char, float> { using type = float; };
template <> struct restype_ex<unsigned char, double> { using type = double; };
template <> struct restype_ex<unsigned char, long long> {
  using type = long long;
};
template <> struct restype_ex<unsigned char, unsigned long long> {
  using type = long long;
};

template <> struct restype_ex<short, char> { using type = int; };
template <> struct restype_ex<short, unsigned char> { using type = int; };
template <> struct restype_ex<short, short> { using type = int; };
template <> struct restype_ex<short, unsigned short> { using type = int; };
template <> struct restype_ex<short, int> { using type = long long; };
template <> struct restype_ex<short, unsigned int> { using type = long long; };
template <> struct restype_ex<short, half> { using type = half; };
template <> struct restype_ex<short, float> { using type = float; };
template <> struct restype_ex<short, double> { using type = double; };
template <> struct restype_ex<short, long long> { using type = long long; };
template <> struct restype_ex<short, unsigned long long> {
  using type = long long;
};

template <> struct restype_ex<unsigned short, char> { using type = int; };
template <> struct restype_ex<unsigned short, unsigned char> {
  using type = int;
};
template <> struct restype_ex<unsigned short, short> { using type = int; };
template <> struct restype_ex<unsigned short, unsigned short> {
  using type = int;
};
template <> struct restype_ex<unsigned short, int> { using type = long long; };
template <> struct restype_ex<unsigned short, unsigned int> {
  using type = long long;
};
template <> struct restype_ex<unsigned short, half> { using type = half; };
template <> struct restype_ex<unsigned short, float> { using type = float; };
template <> struct restype_ex<unsigned short, double> { using type = double; };
template <> struct restype_ex<unsigned short, long long> {
  using type = long long;
};
template <> struct restype_ex<unsigned short, unsigned long long> {
  using type = long long;
};

template <> struct restype_ex<int, char> { using type = long long; };
template <> struct restype_ex<int, unsigned char> { using type = long long; };
template <> struct restype_ex<int, short> { using type = long long; };
template <> struct restype_ex<int, unsigned short> { using type = long long; };
template <> struct restype_ex<int, int> { using type = long long; };
template <> struct restype_ex<int, unsigned int> { using type = long long; };
template <> struct restype_ex<int, half> { using type = half; };
template <> struct restype_ex<int, float> { using type = float; };
template <> struct restype_ex<int, double> { using type = double; };
template <> struct restype_ex<int, long long> { using type = long long; };
template <> struct restype_ex<int, unsigned long long> {
  using type = long long;
};

template <> struct restype_ex<unsigned int, char> { using type = long long; };
template <> struct restype_ex<unsigned int, unsigned char> {
  using type = long long;
};
template <> struct restype_ex<unsigned int, short> { using type = long long; };
template <> struct restype_ex<unsigned int, unsigned short> {
  using type = long long;
};
template <> struct restype_ex<unsigned int, int> { using type = long long; };
template <> struct restype_ex<unsigned int, unsigned int> {
  using type = long long;
};
template <> struct restype_ex<unsigned int, half> { using type = half; };
template <> struct restype_ex<unsigned int, float> { using type = float; };
template <> struct restype_ex<unsigned int, double> { using type = double; };
template <> struct restype_ex<unsigned int, long long> {
  using type = long long;
};
template <> struct restype_ex<unsigned int, unsigned long long> {
  using type = long long;
};

template <> struct restype_ex<half, char> { using type = half; };
template <> struct restype_ex<half, unsigned char> { using type = half; };
template <> struct restype_ex<half, short> { using type = half; };
template <> struct restype_ex<half, unsigned short> { using type = half; };
template <> struct restype_ex<half, int> { using type = half; };
template <> struct restype_ex<half, unsigned int> { using type = half; };
template <> struct restype_ex<half, half> { using type = half; };
template <> struct restype_ex<half, float> { using type = float; };
template <> struct restype_ex<half, double> { using type = double; };
template <> struct restype_ex<half, long long> { using type = half; };
template <> struct restype_ex<half, unsigned long long> { using type = half; };

template <> struct restype_ex<float, char> { using type = float; };
template <> struct restype_ex<float, unsigned char> { using type = float; };
template <> struct restype_ex<float, short> { using type = float; };
template <> struct restype_ex<float, unsigned short> { using type = float; };
template <> struct restype_ex<float, int> { using type = float; };
template <> struct restype_ex<float, unsigned int> { using type = float; };
template <> struct restype_ex<float, half> { using type = float; };
template <> struct restype_ex<float, float> { using type = float; };
template <> struct restype_ex<float, double> { using type = double; };
template <> struct restype_ex<float, long long> { using type = float; };
template <> struct restype_ex<float, unsigned long long> {
  using type = float;
};

template <> struct restype_ex<double, char> { using type = double; };
template <> struct restype_ex<double, unsigned char> { using type = double; };
template <> struct restype_ex<double, short> { using type = double; };
template <> struct restype_ex<double, unsigned short> { using type = double; };
template <> struct restype_ex<double, int> { using type = double; };
template <> struct restype_ex<double, unsigned int> { using type = double; };
template <> struct restype_ex<double, half> { using type = double; };
template <> struct restype_ex<double, float> { using type = double; };
template <> struct restype_ex<double, double> { using type = double; };
template <> struct restype_ex<double, long long> { using type = double; };
template <> struct restype_ex<double, unsigned long long> {
  using type = double;
};

template <> struct restype_ex<long long, char> { using type = long long; };
template <> struct restype_ex<long long, unsigned char> {
  using type = long long;
};
template <> struct restype_ex<long long, short> { using type = long long; };
template <> struct restype_ex<long long, unsigned short> {
  using type = long long;
};
template <> struct restype_ex<long long, int> { using type = long long; };
template <> struct restype_ex<long long, unsigned int> {
  using type = long long;
};
template <> struct restype_ex<long long, half> { using type = half; };
template <> struct restype_ex<long long, float> { using type = float; };
template <> struct restype_ex<long long, double> { using type = double; };
template <> struct restype_ex<long long, long long> { using type = long long; };
template <> struct restype_ex<long long, unsigned long long> {
  using type = long long;
};

template <> struct restype_ex<unsigned long long, char> {
  using type = long long;
};
template <> struct restype_ex<unsigned long long, unsigned char> {
  using type = long long;
};
template <> struct restype_ex<unsigned long long, short> {
  using type = long long;
};
template <> struct restype_ex<unsigned long long, unsigned short> {
  using type = long long;
};
template <> struct restype_ex<unsigned long long, int> {
  using type = long long;
};
template <> struct restype_ex<unsigned long long, unsigned int> {
  using type = long long;
};
template <> struct restype_ex<unsigned long long, half> { using type = half; };
template <> struct restype_ex<unsigned long long, float> {
  using type = float;
};
template <> struct restype_ex<unsigned long long, double> {
  using type = double;
};
template <> struct restype_ex<unsigned long long, long long> {
  using type = long long;
};
template <> struct restype_ex<unsigned long long, unsigned long long> {
  using type = long long;
};

// used in emulation of shl etc operations
template <typename T> struct maxtype { using type = T; };
template <> struct maxtype<char> { using type = int; };
template <> struct maxtype<short> { using type = int; };
template <> struct maxtype<unsigned char> { using type = unsigned int; };
template <> struct maxtype<unsigned short> { using type = unsigned int; };

// used in emulation of abs
template <typename T> struct abstype { using type = T; };
template <> struct abstype<char> { using type = unsigned char; };
template <> struct abstype<short> { using type = unsigned short; };
template <> struct abstype<long long> { using type = unsigned long long; };

template <bool VALUE> struct check_true { static const bool value = false; };
template <> struct check_true<true> { static const bool value = true; };

template <typename T> struct is_inttype { static const bool value = false; };
template <> struct is_inttype<char> { static const bool value = true; };
template <> struct is_inttype<unsigned char> {
  static const bool value = true;
};
template <> struct is_inttype<short> { static const bool value = true; };
template <> struct is_inttype<unsigned short> {
  static const bool value = true;
};
template <> struct is_inttype<int> { static const bool value = true; };
template <> struct is_inttype<unsigned int> { static const bool value = true; };
template <> struct is_inttype<long long> { static const bool value = true; };
template <> struct is_inttype<unsigned long long> {
  static const bool value = true;
};

template <typename T> struct is_byte_type { static const bool value = false; };
template <> struct is_byte_type<char> { static const bool value = true; };
template <> struct is_byte_type<unsigned char> {
  static const bool value = true;
};

template <typename T> struct is_word_type { static const bool value = false; };
template <> struct is_word_type<short> { static const bool value = true; };
template <> struct is_word_type<unsigned short> {
  static const bool value = true;
};

template <typename T> struct is_dword_type { static const bool value = false; };
template <> struct is_dword_type<int> { static const bool value = true; };
template <> struct is_dword_type<unsigned int> {
  static const bool value = true;
};

template <typename T> struct is_qf_type { static const bool value = false; };
template <> struct is_qf_type<unsigned char> {
  static const bool value = true;
};

template <typename T> struct is_hf_type { static const bool value = false; };
template <> struct is_hf_type<half> { static const bool value = true; };

template <typename T> struct is_fp_type { static const bool value = false; };
template <> struct is_fp_type<float> { static const bool value = true; };

template <typename T> struct is_df_type { static const bool value = false; };
template <> struct is_df_type<double> { static const bool value = true; };

template <typename T> struct is_fp_or_dword_type {
  static const bool value = false;
};
template <> struct is_fp_or_dword_type<int> { static const bool value = true; };
template <> struct is_fp_or_dword_type<unsigned int> {
  static const bool value = true;
};
template <> struct is_fp_or_dword_type<float> {
  static const bool value = true;
};
// The check is only used for dataport APIs,
// which also support df data type.
template <> struct is_fp_or_dword_type<double> {
  static const bool value = true;
};

template <typename T> struct is_ushort_type {
  static const bool value = false;
};
template <> struct is_ushort_type<unsigned short> {
  static const bool value = true;
};

template <typename T1, typename T2> struct is_float_dword {
  static const bool value = false;
};
template <> struct is_float_dword<float, int> {
  static const bool value = true;
};
template <> struct is_float_dword<float, unsigned int> {
  static const bool value = true;
};
template <> struct is_float_dword<int, float> {
  static const bool value = true;
};
template <> struct is_float_dword<unsigned int, float> {
  static const bool value = true;
};

template <typename T> struct hftype { static const bool value = false; };
template <> struct hftype<half> { static const bool value = true; };

template <typename T> struct fptype { static const bool value = false; };
template <> struct fptype<float> { static const bool value = true; };

template <typename T> struct dftype { static const bool value = false; };
template <> struct dftype<double> { static const bool value = true; };

template <typename T> struct esimdtype;
template <> struct esimdtype<char> { static const bool value = true; };

template <> struct esimdtype<signed char> { static const bool value = true; };

template <> struct esimdtype<unsigned char> { static const bool value = true; };

template <> struct esimdtype<short> { static const bool value = true; };

template <> struct esimdtype<unsigned short> {
  static const bool value = true;
};
template <> struct esimdtype<int> { static const bool value = true; };

template <> struct esimdtype<unsigned int> { static const bool value = true; };

template <> struct esimdtype<unsigned long> { static const bool value = true; };

template <> struct esimdtype<half> { static const bool value = true; };

template <> struct esimdtype<float> { static const bool value = true; };

template <> struct esimdtype<double> { static const bool value = true; };

template <> struct esimdtype<long long> { static const bool value = true; };

template <> struct esimdtype<unsigned long long> {
  static const bool value = true;
};

template <typename T> struct bytetype;
template <> struct bytetype<char> { static const bool value = true; };
template <> struct bytetype<unsigned char> { static const bool value = true; };

template <typename T> struct wordtype;
template <> struct wordtype<short> { static const bool value = true; };
template <> struct wordtype<unsigned short> { static const bool value = true; };

template <typename T> struct dwordtype;
template <> struct dwordtype<int> { static const bool value = true; };
template <> struct dwordtype<unsigned int> { static const bool value = true; };

} // namespace detail
} // namespace emu
} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#endif // #ifndef __SYCL_DEVICE_ONLY__
