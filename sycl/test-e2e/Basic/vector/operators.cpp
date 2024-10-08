// RUN: %{build} -fsycl-device-code-split=per_kernel -DSYCL2020_DISABLE_DEPRECATION_WARNINGS -o %t.out
// RUN: %{run} %t.out

//==---------- vector_operators.cpp - SYCL vec<> operators test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL_SIMPLE_SWIZZLES
#include <sycl/detail/core.hpp>
#include <sycl/types.hpp>
namespace s = sycl;

template <typename ResultVecT>
void check_result_length_4(ResultVecT &res, ResultVecT &expected_res) {
  assert(res.get_count() == 4 && expected_res.get_count() == 4);

  assert(static_cast<bool>(res.template swizzle<s::elem::s0>() ==
                           expected_res.template swizzle<s::elem::s0>()));
  assert(static_cast<bool>(res.template swizzle<s::elem::s1>() ==
                           expected_res.template swizzle<s::elem::s1>()));
  assert(static_cast<bool>(res.template swizzle<s::elem::s2>() ==
                           expected_res.template swizzle<s::elem::s2>()));
  assert(static_cast<bool>(res.template swizzle<s::elem::s3>() ==
                           expected_res.template swizzle<s::elem::s3>()));
}

template <typename T, int N> void check_vector_size() {
  using vec_type = s::vec<T, N>;
  vec_type Vec;
  constexpr auto length = (N == 3 ? 4 : N);
  assert(Vec.size() == N);
  assert(Vec.byte_size() == sizeof(T) * length);
  static_assert(noexcept(Vec.byte_size()));
}

int main() {
  s::queue Queue;

  /* Separate checks for NumElements=1 edge case */

  {
    using vec_type = s::vec<char, 1>;
    vec_type res;
    {
      s::buffer<vec_type, 1> Buf(&res, s::range<1>(1));
      Queue.submit([&](s::handler &cgh) {
        auto Acc = Buf.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isequal_vec_op_1_elem>([=]() {
          vec_type vec1(2);
          vec_type vec2(2);
          Acc[0] = vec1 == vec2;
        });
      });
    }
    // 1-element vector operators follow vector 0/-1 logic
    vec_type expected_res(-1);
    assert(static_cast<bool>(res.template swizzle<sycl::elem::s0>() ==
                             expected_res.template swizzle<sycl::elem::s0>()));
  }

  {
    using vec_type = s::vec<char, 1>;
    vec_type res;
    {
      s::buffer<vec_type, 1> Buf(&res, s::range<1>(1));
      Queue.submit([&](s::handler &cgh) {
        auto Acc = Buf.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isequal_vec_op_1_elem_scalar>([=]() {
          vec_type vec(2);
          char rhs_scalar = 2;
          Acc[0] = vec == rhs_scalar;
        });
      });
    }
    // 1-element vector operators follow vector 0/-1 logic
    vec_type expected_res(-1);
    assert(static_cast<bool>(res.template swizzle<sycl::elem::s0>() ==
                             expected_res.template swizzle<sycl::elem::s0>()));
  }

  /* Test different operators, different types
   *  for length 4 */

  // SYCL vec<>, SYCL vec<>

  // Operator ==, unsigned int
  {
    using res_vec_type = s::int4;
    res_vec_type res;
    {
      s::buffer<res_vec_type, 1> Buf(&res, s::range<1>(1));
      Queue.submit([&](s::handler &cgh) {
        auto Acc = Buf.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isequal_vec_op>([=]() {
          s::uint4 vec1(42, 42, 42, 0);
          s::uint4 vec2(0, 42, 42, 0);
          Acc[0] = vec1 == vec2;
        });
      });
    }
    res_vec_type expected(0, -1, -1, -1);
    check_result_length_4<res_vec_type>(res, expected);
  }

  // Operator <, double
  if (Queue.get_device().has(sycl::aspect::fp64)) {
    using res_vec_type = s::long4;
    res_vec_type res;
    {
      s::buffer<res_vec_type, 1> Buf(&res, s::range<1>(1));
      Queue.submit([&](s::handler &cgh) {
        auto Acc = Buf.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isless_vec_op>([=]() {
          s::double4 vec1(0.5, 10.1, 10.2, 10.3);
          s::double4 vec2(10.5, 0.1, 0.2, 0.3);
          Acc[0] = vec1 < vec2;
        });
      });
    }
    res_vec_type expected(-1, 0, 0, 0);
    check_result_length_4<res_vec_type>(res, expected);
  }

  // Operator >, char
  {
    using res_vec_type = s::char4;
    res_vec_type res;
    {
      s::buffer<res_vec_type, 1> Buf(&res, s::range<1>(1));
      Queue.submit([&](s::handler &cgh) {
        auto Acc = Buf.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isgreater_vec_op>([=]() {
          s::char4 vec1(0, 0, 42, 42);
          s::char4 vec2(42, 0, 0, -42);
          Acc[0] = vec1 > vec2;
        });
      });
    }
    res_vec_type expected(0, 0, -1, -1);
    check_result_length_4<res_vec_type>(res, expected);
  }

  // Operator <=, half
  if (Queue.get_device().has(sycl::aspect::fp16)) {
    using res_vec_type = s::short4;
    res_vec_type res;
    {
      s::buffer<res_vec_type, 1> Buf(&res, s::range<1>(1));
      Queue.submit([&](s::handler &cgh) {
        auto Acc = Buf.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnotgreater_vec_op>([=]() {
          s::half4 vec1(0, 0, 42, 42);
          s::half4 vec2(42, 0, 0, -42);
          Acc[0] = vec1 <= vec2;
        });
      });
    }
    res_vec_type expected(-1, -1, 0, 0);
    check_result_length_4<res_vec_type>(res, expected);
  }

  // SYCL vec<>, OpenCL built-in

  // Operator >=, unsigned long
  {
    using res_vec_type = s::long4;
    res_vec_type res;
    {
      s::buffer<res_vec_type, 1> Buf(&res, s::range<1>(1));
      Queue.submit([&](s::handler &cgh) {
        auto Acc = Buf.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnotless_vec_op>([=]() {
          s::ulong4 vec1(0, 0, 42, 42);
          s::ulong4 vec2{42, 0, 0, 0};
          Acc[0] = vec1 >= vec2;
        });
      });
    }
    res_vec_type expected(0, -1, -1, -1);
    check_result_length_4<res_vec_type>(res, expected);
  }

  // Operator !=, unsigned short
  {
    using res_vec_type = s::short4;
    res_vec_type res;
    {
      s::buffer<res_vec_type, 1> Buf(&res, s::range<1>(1));
      Queue.submit([&](s::handler &cgh) {
        auto Acc = Buf.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnotequal_vec_op>([=]() {
          s::ushort4 vec1(0, 0, 42, 42);
          s::ushort4 vec2{42, 0, 0, 42};
          Acc[0] = vec1 != vec2;
        });
      });
    }
    res_vec_type expected(-1, 0, -1, 0);
    check_result_length_4<res_vec_type>(res, expected);
  }

  // Operator &&, int
  {
    using res_vec_type = s::int4;
    res_vec_type res;
    {
      s::buffer<res_vec_type, 1> Buf(&res, s::range<1>(1));
      Queue.submit([&](s::handler &cgh) {
        auto Acc = Buf.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class logical_and_vec_op>([=]() {
          s::int4 vec1(0, 0, 42, 42);
          s::int4 vec2{42, 0, 0, 42};
          Acc[0] = vec1 && vec2;
        });
      });
    }
    res_vec_type expected(0, 0, 0, -1);
    check_result_length_4<res_vec_type>(res, expected);
  }

  // Operator ||, int
  {
    using res_vec_type = s::int4;
    res_vec_type res;
    {
      s::buffer<res_vec_type, 1> Buf(&res, s::range<1>(1));
      Queue.submit([&](s::handler &cgh) {
        auto Acc = Buf.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class logical_or_vec_op>([=]() {
          s::int4 vec1(0, 0, 42, 42);
          s::int4 vec2{42, 0, 0, 42};
          Acc[0] = vec1 || vec2;
        });
      });
    }
    res_vec_type expected(-1, 0, -1, -1);
    check_result_length_4<res_vec_type>(res, expected);
  }

  // as() function.
  // reinterprets each element as a different datatype.
  {
    using res_vec_type = s::int4;
    res_vec_type res;
    {
      s::buffer<res_vec_type, 1> Buf(&res, s::range<1>(1));
      Queue.submit([&](s::handler &cgh) {
        auto Acc = Buf.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class as_op>([=]() {
          s::float4 vec1(4.5f, 0, 3.5f, -10.0f);
          Acc[0] = vec1.template as<res_vec_type>();
        });
      });
    }
    res_vec_type expected(1083179008 /*0x40900000*/, 0,
                          1080033280 /*0x40600000*/,
                          -1054867456 /*0xc1200000*/);
    check_result_length_4<res_vec_type>(res, expected);
  }

  // size() and byte_size() functions
  {
    check_vector_size<char, 1>();
    check_vector_size<int, 2>();
    check_vector_size<float, 3>();
    check_vector_size<double, 4>();
  }

  return 0;
}
