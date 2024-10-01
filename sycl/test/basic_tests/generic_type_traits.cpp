// RUN: %clangxx -fsycl -fsyntax-only %s

#include <cassert>
#include <iostream>
#include <sycl/detail/common.hpp>
#include <sycl/half_type.hpp>
#include <sycl/sycl.hpp>

namespace s = sycl;
namespace d = sycl::detail;

using i_t = int;
using f_t = float;

namespace t {
using c_t = char;
using d_t = double;
} // namespace t

struct v {};

int main() {
  static_assert(d::is_genfloat_v<s::opencl::cl_float> == true);
  static_assert(d::is_genfloat_v<s::vec<s::opencl::cl_float, 4>> == true);

  static_assert(d::is_half_v<s::half>);

  static_assert(d::is_bfloat16_v<sycl::ext::oneapi::bfloat16>);
  static_assert(d::is_half_or_bf16_v<s::half>);
  static_assert(d::is_half_or_bf16_v<sycl::ext::oneapi::bfloat16>);

  // TODO add checks for the following type traits
  /*
  is_doublen
  is_genfloatd

  is_halfn
  is_genfloath

  is_genfloat

  is_sgenfloat
  is_vgenfloat

  is_charn
  is_scharn
  is_ucharn
  is_igenchar
  is_ugenchar
  is_genchar

  is_shortn
  is_genshort
  is_ushortn
  is_ugenshort

  is_uintn
  is_genint
  */

  /*
  is_ulonglongn
  is_ugenlonglong
  is_longlongn
  is_genlonglong

  is_igenlonginteger
  is_ugenlonginteger

  is_geninteger
  is_sgeninteger


  is_sigeninteger
  is_sugeninteger

  unsing_integeral_to_float_point
  float_point_to_sign_integeral

  make_unsigned
  make_larger
  */

  // checks for some type conversions.
  static_assert(std::is_same_v<d::ConvertToOpenCLType_t<s::opencl::cl_int>,
                               s::opencl::cl_int>);

#ifdef __SYCL_DEVICE_ONLY__
  static_assert(
      std::is_same_v<d::ConvertToOpenCLType_t<s::vec<s::opencl::cl_int, 2>>,
                     s::opencl::cl_int __attribute__((ext_vector_type(2)))>);

  static_assert(std::is_same_v<
                d::ConvertToOpenCLType_t<s::multi_ptr<
                    s::opencl::cl_int, s::access::address_space::global_space,
                    s::access::decorated::yes>>,
                __attribute__((opencl_global)) s::opencl::cl_int *>);

  using int_vec2 = s::opencl::cl_int __attribute__((ext_vector_type(2)));
  static_assert(
      std::is_same_v<d::ConvertToOpenCLType_t<
                         s::multi_ptr<s::vec<s::opencl::cl_int, 2>,
                                      s::access::address_space::global_space,
                                      s::access::decorated::yes>>,
                     __attribute__((opencl_global)) int_vec2 *>);

  static_assert(
      std::is_same_v<d::ConvertToOpenCLType_t<long long>, s::opencl::cl_long>);

  static_assert(
      std::is_same_v<d::ConvertToOpenCLType_t<s::vec<long long, 2>>,
                     s::opencl::cl_long __attribute__((ext_vector_type(2)))>);

  static_assert(
      std::is_same_v<d::ConvertToOpenCLType_t<s::multi_ptr<
                         long long, s::access::address_space::global_space,
                         s::access::decorated::yes>>,
                     __attribute__((opencl_global)) s::opencl::cl_long *>);

  using long_vec2 = s::opencl::cl_long __attribute__((ext_vector_type(2)));
  static_assert(
      std::is_same_v<
          d::ConvertToOpenCLType_t<s::multi_ptr<
              s::vec<long long, 2>, s::access::address_space::global_space,
              s::access::decorated::yes>>,
          __attribute__((opencl_global)) long_vec2 *>);

  using signed_char2 = s::opencl::cl_char __attribute__((ext_vector_type(2)));
  static_assert(std::is_same_v<
                d::ConvertToOpenCLType_t<s::multi_ptr<
                    s::vec<char, 2>, s::access::address_space::global_space,
                    s::access::decorated::yes>>,
          __attribute__((opencl_global)) signed_char2 *>);
  static_assert(
      std::is_same_v<
          d::ConvertToOpenCLType_t<s::multi_ptr<
              s::vec<signed char, 2>, s::access::address_space::global_space,
              s::access::decorated::yes>>,
          __attribute__((opencl_global)) signed_char2 *>);

#endif

#ifdef __SYCL_DEVICE_ONLY__
  static_assert(
      std::is_same_v<d::ConvertToOpenCLType_t<s::vec<s::opencl::cl_int, 2>>,
                     s::vec<s::opencl::cl_int, 2>::vector_t>);
  static_assert(std::is_same_v<d::ConvertToOpenCLType_t<s::vec<long long, 2>>,
                               s::vec<s::opencl::cl_long, 2>::vector_t>);
  static_assert(std::is_same_v<
                d::ConvertToOpenCLType_t<s::multi_ptr<
                    s::opencl::cl_int, s::access::address_space::global_space,
                    s::access::decorated::yes>>,
                s::multi_ptr<s::opencl::cl_int,
                             s::access::address_space::global_space,
                             s::access::decorated::yes>::pointer>);
  static_assert(
      std::is_same_v<d::ConvertToOpenCLType_t<
                         s::multi_ptr<s::vec<s::opencl::cl_int, 4>,
                                      s::access::address_space::global_space,
                                      s::access::decorated::yes>>,
                     s::multi_ptr<s::vec<s::opencl::cl_int, 4>::vector_t,
                                  s::access::address_space::global_space,
                                  s::access::decorated::yes>::pointer>);
#endif
  static_assert(std::is_same_v<d::ConvertToOpenCLType_t<s::half>,
                               d::half_impl::BIsRepresentationT>);

  s::multi_ptr<int, s::access::address_space::global_space,
               s::access::decorated::yes>
      mp;
  int *dp = mp;
}
