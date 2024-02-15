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
  // is_floatn
  static_assert(d::is_floatn_v<s::vec<s::opencl::cl_float, 4>> == true);
  static_assert(d::is_floatn_v<s::vec<s::opencl::cl_float, 16>> == true);
  static_assert(d::is_floatn_v<s::float4> == true, "");
  static_assert(d::is_floatn_v<s::float16> == true, "");

  static_assert(d::is_floatn_v<s::opencl::cl_float> == false);
  static_assert(d::is_floatn_v<s::opencl::cl_int> == false);
  static_assert(d::is_floatn_v<i_t> == false, "");
  static_assert(d::is_floatn_v<f_t> == false, "");
  static_assert(d::is_floatn_v<t::c_t> == false, "");
  static_assert(d::is_floatn_v<t::d_t> == false, "");
  static_assert(d::is_floatn_v<v> == false, "");
  // is_genfloatf
  static_assert(d::is_genfloatf_v<s::vec<s::opencl::cl_float, 4>> == true);
  static_assert(d::is_genfloatf_v<s::vec<s::opencl::cl_float, 16>> == true);
  static_assert(d::is_genfloatf_v<s::opencl::cl_float> == true);
  static_assert(d::is_genfloatf_v<s::float4> == true);
  static_assert(d::is_genfloatf_v<s::float16> == true);
  static_assert(d::is_genfloatf_v<f_t> == true, "");

  static_assert(d::is_genfloatf_v<s::opencl::cl_int> == false);
  static_assert(d::is_genfloatf_v<i_t> == false, "");
  static_assert(d::is_genfloatf_v<t::c_t> == false, "");
  static_assert(d::is_genfloatf_v<t::d_t> == false, "");
  static_assert(d::is_genfloatf_v<v> == false, "");

  //

  static_assert(d::is_genfloat_v<s::opencl::cl_float> == true);
  static_assert(d::is_genfloat_v<s::vec<s::opencl::cl_float, 4>> == true);

  static_assert(d::is_ugenint_v<s::vec<s::opencl::cl_float, 4>> == false);
  static_assert(d::is_ugenint_v<s::float4> == false);

  static_assert(d::is_ugenint_v<s::opencl::cl_uint> == true);
  static_assert(d::is_ugenint_v<unsigned int> == true);

  static_assert(d::is_ugenint_v<s::vec<s::opencl::cl_uint, 3>> == true);
  static_assert(d::is_ugenint_v<s::uint3> == true);

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

  is_gengeofloat
  is_gengeodouble
  is_gengeohalf

  is_vgengeofloat
  is_vgengeodouble
  is_vgengeohalf

  is_gencrossfloat
  is_gencrossdouble
  is_gencrosshalf
  is_gencross

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
  is_ugenint
  is_intn
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
  is_igeninteger
  is_ugeninteger
  is_sgeninteger
  is_vgeninteger


  is_sigeninteger
  is_sugeninteger
  is_vigeninteger
  is_vugeninteger

  is_gentype

  is_igeninteger8bit
  is_igeninteger16bit
  is_igeninteger32bit
  is_igeninteger64bit

  is_ugeninteger8bit
  is_ugeninteger16bit
  is_ugeninteger32bit
  is_ugeninteger64bit

  is_genintptr
  is_genfloatptr

  unsing_integeral_to_float_point
  */
  // is_nan_type
  static_assert(d::is_nan_type_v<unsigned long long int> == true, "");
  static_assert(d::is_nan_type_v<long long> == false, "");
  static_assert(d::is_nan_type_v<unsigned long long> == true, "");
  static_assert(d::is_nan_type_v<unsigned long> == true, "");
  static_assert(d::is_nan_type_v<long> == false, "");
  static_assert(d::is_nan_type_v<unsigned long> == true, "");
  /*
  float_point_to_sign_integeral

  make_unsigned
  make_larger
  */

  // checks for some type conversions.
  static_assert(std::is_same<d::SelectMatchingOpenCLType_t<s::opencl::cl_int>,
                             s::opencl::cl_int>::value);

  static_assert(
      std::is_same<d::SelectMatchingOpenCLType_t<s::vec<s::opencl::cl_int, 2>>,
                   s::vec<s::opencl::cl_int, 2>>::value);

  static_assert(std::is_same<
                d::SelectMatchingOpenCLType_t<s::multi_ptr<
                    s::opencl::cl_int, s::access::address_space::global_space,
                    s::access::decorated::yes>>,
                s::multi_ptr<s::opencl::cl_int,
                             s::access::address_space::global_space,
                             s::access::decorated::yes>>::value);

  static_assert(
      std::is_same<d::SelectMatchingOpenCLType_t<
                       s::multi_ptr<s::vec<s::opencl::cl_int, 2>,
                                    s::access::address_space::global_space,
                                    s::access::decorated::yes>>,
                   s::multi_ptr<s::vec<s::opencl::cl_int, 2>,
                                s::access::address_space::global_space,
                                s::access::decorated::yes>>::value);

  static_assert(std::is_same<d::SelectMatchingOpenCLType_t<long long>,
                             s::opencl::cl_long>::value);

  static_assert(
      std::is_same<d::SelectMatchingOpenCLType_t<s::vec<long long, 2>>,
                   s::vec<s::opencl::cl_long, 2>>::value);

  static_assert(
      std::is_same<d::SelectMatchingOpenCLType_t<s::multi_ptr<
                       long long, s::access::address_space::global_space,
                       s::access::decorated::yes>>,
                   s::multi_ptr<s::opencl::cl_long,
                                s::access::address_space::global_space,
                                s::access::decorated::yes>>::value);

  static_assert(
      std::is_same<
          d::SelectMatchingOpenCLType_t<s::multi_ptr<
              s::vec<long long, 2>, s::access::address_space::global_space,
              s::access::decorated::yes>>,
          s::multi_ptr<s::vec<s::opencl::cl_long, 2>,
                       s::access::address_space::global_space,
                       s::access::decorated::yes>>::value);

#ifdef __SYCL_DEVICE_ONLY__
  static_assert(
      std::is_same<d::ConvertToOpenCLType_t<s::vec<s::opencl::cl_int, 2>>,
                   s::vec<s::opencl::cl_int, 2>::vector_t>::value);
  static_assert(std::is_same<d::ConvertToOpenCLType_t<s::vec<long long, 2>>,
                             s::vec<s::opencl::cl_long, 2>::vector_t>::value);
  static_assert(std::is_same<
                d::ConvertToOpenCLType_t<s::multi_ptr<
                    s::opencl::cl_int, s::access::address_space::global_space,
                    s::access::decorated::yes>>,
                s::multi_ptr<s::opencl::cl_int,
                             s::access::address_space::global_space,
                             s::access::decorated::yes>::pointer>::value);
  static_assert(
      std::is_same<d::ConvertToOpenCLType_t<
                       s::multi_ptr<s::vec<s::opencl::cl_int, 4>,
                                    s::access::address_space::global_space,
                                    s::access::decorated::yes>>,
                   s::multi_ptr<s::vec<s::opencl::cl_int, 4>::vector_t,
                                s::access::address_space::global_space,
                                s::access::decorated::yes>::pointer>::value);
#endif
  static_assert(std::is_same<d::ConvertToOpenCLType_t<s::half>,
                             d::half_impl::BIsRepresentationT>::value,
                "");

  s::multi_ptr<int, s::access::address_space::global_space,
               s::access::decorated::yes>
      mp;
  int *dp = mp;
}
