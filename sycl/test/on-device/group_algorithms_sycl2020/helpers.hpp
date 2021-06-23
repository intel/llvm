//==---------- helpers.hpp -*- C++ -*--------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <bitset>
#include <cmath>
#include <complex>
#include <iostream>

using namespace cl::sycl;

// ---- utils
template <typename T1, int N> struct utils {
  static T1 add_vec(const vec<T1, N> &v);
  static bool cmp_vec(const vec<T1, N> &v, const vec<T1, N> &r);
  static std::string stringify_vec(const vec<T1, N> &v);
};
template <typename T2> struct utils<T2, 1> {
  static T2 add_vec(const vec<T2, 1> &v) { return v.s0(); }
  static bool cmp_vec(const vec<T2, 1> &v, const vec<T2, 1> &r) {
    return v.s0() == r.s0();
  }
  static std::string stringify_vec(const vec<T2, 1> &v) {
    return std::to_string((T2)v.s0());
  }
};
template <typename T2> struct utils<T2, 2> {
  static T2 add_vec(const vec<T2, 2> &v) { return v.s0() + v.s1(); }
  static bool cmp_vec(const vec<T2, 2> &v, const vec<T2, 2> &r) {
    return v.s0() == r.s0() && v.s1() == r.s1();
  }
  static std::string stringify_vec(const vec<T2, 2> &v) {
    return std::string("(") + std::to_string((T2)v.s0()) + ", " +
           std::to_string((T2)v.s1()) + " )";
  }
};
template <typename T2> struct utils<T2, 3> {
  static T2 add_vec(const vec<T2, 3> &v) { return v.s0() + v.s1() + v.s2(); }
  static bool cmp_vec(const vec<T2, 3> &v, const vec<T2, 3> &r) {
    return v.s0() == r.s0() && v.s1() == r.s1() && v.s2() == r.s2();
  }
  static std::string stringify_vec(const vec<T2, 2> &v) {
    return std::string("(") + std::to_string((T2)v.s0()) + ", " +
           std::to_string((T2)v.s1()) + ", " + std::to_string((T2)v.s3()) +
           " )";
  }
};
template <typename T2> struct utils<T2, 4> {
  static T2 add_vec(const vec<T2, 4> &v) {
    return v.s0() + v.s1() + v.s2() + v.s3();
  }
  static bool cmp_vec(const vec<T2, 4> &v, const vec<T2, 4> &r) {
    return v.s0() == r.s0() && v.s1() == r.s1() && v.s2() == r.s2() &&
           v.s3() == r.s3();
  }
  static std::string stringify_vec(const vec<T2, 4> &v) {
    return std::string("(") + std::to_string((T2)v.s0()) + ", " +
           std::to_string((T2)v.s1()) + std::to_string((T2)v.s2()) + ", " +
           std::to_string((T2)v.s3()) + " )";
  }
};
template <typename T2> struct utils<T2, 8> {
  static T2 add_vec(const vec<T2, 8> &v) {
    return v.s0() + v.s1() + v.s2() + v.s3() + v.s4() + v.s5() + v.s6() +
           v.s7();
  }
  static bool cmp_vec(const vec<T2, 8> &v, const vec<T2, 8> &r) {
    return v.s0() == r.s0() && v.s1() == r.s1() && v.s2() == r.s2() &&
           v.s3() == r.s3() && v.s4() == r.s4() && v.s5() == r.s5() &&
           v.s6() == r.s6() && v.s7() == r.s7();
  }
  static std::string stringify_vec(const vec<T2, 8> &v) {
    return std::string("(") + std::to_string((T2)v.s0()) + ", " +
           std::to_string((T2)v.s1()) + std::to_string((T2)v.s2()) + ", " +
           std::to_string((T2)v.s3()) + std::to_string((T2)v.s4()) + ", " +
           std::to_string((T2)v.s5()) + std::to_string((T2)v.s6()) + ", " +
           std::to_string((T2)v.s7()) + " )";
  }
};

template <typename T2> struct utils<T2, 16> {
  static T2 add_vec(const vec<T2, 16> &v) {
    return v.s0() + v.s1() + v.s2() + v.s3() + v.s4() + v.s5() + v.s6() +
           v.s7() + v.s8() + v.s9() + v.sA() + v.sB() + v.sC() + v.sD() +
           v.sE() + v.sF();
  }
  static bool cmp_vec(const vec<T2, 16> &v, const vec<T2, 16> &r) {
    return v.s0() == r.s0() && v.s1() == r.s1() && v.s2() == r.s2() &&
           v.s3() == r.s3() && v.s4() == r.s4() && v.s5() == r.s5() &&
           v.s6() == r.s6() && v.s7() == r.s7() && v.s8() == r.s8() &&
           v.s9() == r.s9() && v.sA() == r.sA() && v.sB() == r.sB() &&
           v.sC() == r.sC() && v.sD() == r.sD() && v.sE() == r.sE() &&
           v.sF() == r.sF();
  }
  static std::string stringify_vec(const vec<T2, 16> &v) {
    return std::string("(") + std::to_string((T2)v.s0()) + ", " +
           std::to_string((T2)v.s1()) + std::to_string((T2)v.s2()) + ", " +
           std::to_string((T2)v.s3()) + std::to_string((T2)v.s4()) + ", " +
           std::to_string((T2)v.s5()) + std::to_string((T2)v.s6()) + ", " +
           std::to_string((T2)v.s7()) + std::to_string((T2)v.s8()) + ", " +
           std::to_string((T2)v.s9()) + std::to_string((T2)v.sA()) + ", " +
           std::to_string((T2)v.sB()) + std::to_string((T2)v.sC()) + ", " +
           std::to_string((T2)v.sE()) + std::to_string((T2)v.sD()) + ", " +
           std::to_string((T2)v.sF()) + " )";
  }
};

// ---- exit_if_not_equal
template <typename T> void exit_if_not_equal(T val, T ref, const char *name) {
  if (std::is_floating_point<T>::value) {
    auto cmp_val = std::bitset<CHAR_BIT * sizeof(T)>(val);
    auto cmp_ref = std::bitset<CHAR_BIT * sizeof(T)>(ref);
    if (cmp_val != cmp_ref) {
      std::cout << "Unexpected result for " << name << ": " << val << "("
                << cmp_val << ") expected value: " << ref << "(" << cmp_ref
                << ")" << std::endl;
      exit(1);
    }
  } else {
    if ((val - ref) != 0) {
      std::cout << "Unexpected result for " << name << ": " << (long)val
                << " expected value: " << (long)ref << std::endl;
      exit(1);
    }
  }
}

// template <typename T>
// void exit_if_not_equal(std::complex<T> val, std::complex<T> ref,
//                        const char *name) {
//   std::string Name{name};
//   exit_if_not_equal(val.real(), ref.real(), (Name + ".real()").c_str());
//   exit_if_not_equal(val.imag(), ref.imag(), (Name + ".imag()").c_str());
// }

template <typename T> void exit_if_not_equal(T *val, T *ref, const char *name) {
  if ((val - ref) != 0) {
    std::cout << "Unexpected result for " << name << ": " << val
              << " expected value: " << ref << std::endl;
    exit(1);
  }
}

template <> void exit_if_not_equal(half val, half ref, const char *name) {
  int16_t cmp_val = reinterpret_cast<int16_t &>(val);
  int16_t cmp_ref = reinterpret_cast<int16_t &>(ref);
  if (std::abs(cmp_val - cmp_ref) > 1) {
    std::cout << "Unexpected result for " << name << ": " << (float)val
              << " expected value: " << (float)ref << std::endl;
    exit(1);
  }
}

template <typename T, int N>
void exit_if_not_equal_vec(vec<T, N> val, vec<T, N> ref, const char *name) {
  if (!utils<T, N>::cmp_vec(ref, val)) {
    std::cout << "Unexpected result for " << name << ": "
              << utils<T, N>::stringify_vec(val)
              << " expected value: " << utils<T, N>::stringify_vec(ref)
              << std::endl;

    exit(1);
  }
}