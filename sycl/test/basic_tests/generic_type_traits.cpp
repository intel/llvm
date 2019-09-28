// RUN: %clangxx -fsycl %s -o %t.out

#include <CL/sycl.hpp>
#include <CL/sycl/detail/common.hpp>
#include <cassert>
#include <iostream>

namespace s = cl::sycl;
namespace d = cl::sycl::detail;

using i_t = int;
using f_t = float;

namespace t {
using c_t = char;
using d_t = double;
} // namespace t

struct v {};

int main() {
  // is_floatn
  static_assert(d::is_floatn<s::cl_float4>::value == true, "");
  static_assert(d::is_floatn<s::cl_float16>::value == true, "");

  static_assert(d::is_floatn<s::cl_float>::value == false, "");
  static_assert(d::is_floatn<s::cl_int>::value == false, "");
  static_assert(d::is_floatn<i_t>::value == false, "");
  static_assert(d::is_floatn<f_t>::value == false, "");
  static_assert(d::is_floatn<t::c_t>::value == false, "");
  static_assert(d::is_floatn<t::d_t>::value == false, "");
  static_assert(d::is_floatn<v>::value == false, "");
  // is_genfloatf
  static_assert(d::is_genfloatf<s::cl_float4>::value == true, "");
  static_assert(d::is_genfloatf<s::cl_float16>::value == true, "");
  static_assert(d::is_genfloatf<s::cl_float>::value == true, "");
  static_assert(d::is_genfloatf<f_t>::value == true, "");

  static_assert(d::is_genfloatf<s::cl_int>::value == false, "");
  static_assert(d::is_genfloatf<i_t>::value == false, "");
  static_assert(d::is_genfloatf<t::c_t>::value == false, "");
  static_assert(d::is_genfloatf<t::d_t>::value == false, "");
  static_assert(d::is_genfloatf<v>::value == false, "");

  //

  static_assert(d::is_genfloat<s::cl_float>::value == true, "");
  static_assert(d::is_genfloat<s::cl_float4>::value == true, "");
  static_assert(d::is_genfloat<s::cl_float4>::value == true, "");

  static_assert(d::is_ugenint<s::cl_float4>::value == false, "");

  static_assert(d::is_ugenint<s::cl_uint>::value == true, "");

  static_assert(d::is_ugenint<s::cl_uint3>::value == true, "");

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

  // is_ulongn
  static_assert(d::is_ulongn<s::uchar>::value == false, "");
  static_assert(d::is_ulongn<s::ulong>::value == false, "");
  static_assert(d::is_ulongn<s::ulong2>::value == true, "");
  static_assert(d::is_ulongn<s::ulong3>::value == true, "");
  static_assert(d::is_ulongn<s::ulong4>::value == true, "");
  static_assert(d::is_ulongn<s::ulong8>::value == true, "");
  static_assert(d::is_ulongn<s::ulong16>::value == true, "");

  // is_ugenlong
  static_assert(d::is_ugenlong<s::uchar>::value == false, "");
  static_assert(d::is_ugenlong<s::ulong>::value == true, "");
  static_assert(d::is_ugenlong<s::ulong2>::value == true, "");
  static_assert(d::is_ugenlong<s::ulong3>::value == true, "");
  static_assert(d::is_ugenlong<s::ulong4>::value == true, "");
  static_assert(d::is_ugenlong<s::ulong8>::value == true, "");
  static_assert(d::is_ugenlong<s::ulong16>::value == true, "");

  // is_longn
  static_assert(d::is_longn<char>::value == false, "");
  static_assert(d::is_longn<long>::value == false, "");
  static_assert(d::is_longn<s::long2>::value == true, "");
  static_assert(d::is_longn<s::long3>::value == true, "");
  static_assert(d::is_longn<s::long4>::value == true, "");
  static_assert(d::is_longn<s::long8>::value == true, "");
  static_assert(d::is_longn<s::long16>::value == true, "");

  // is_genlong
  static_assert(d::is_genlong<char>::value == false, "");
  static_assert(d::is_genlong<long>::value == true, "");
  static_assert(d::is_genlong<s::long2>::value == true, "");
  static_assert(d::is_genlong<s::long3>::value == true, "");
  static_assert(d::is_genlong<s::long4>::value == true, "");
  static_assert(d::is_genlong<s::long8>::value == true, "");
  static_assert(d::is_genlong<s::long16>::value == true, "");

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
  static_assert(d::is_nan_type<unsigned long long int>::value == true, "");
  static_assert(d::is_nan_type<s::longlong>::value == false, "");
  static_assert(d::is_nan_type<s::ulonglong>::value == true, "");
  static_assert(d::is_nan_type<unsigned long>::value == true, "");
  static_assert(d::is_nan_type<long>::value == false, "");
  static_assert(d::is_nan_type<s::ulong>::value == true, "");
  /*
  float_point_to_sign_integeral

  make_unsigned
  make_larger
  */

  // checks for some type conversions.
  static_assert(
      std::is_same<d::SelectMatchingOpenCLType_t<s::cl_int>, s::cl_int>::value,
      "");

  static_assert(
      std::is_same<d::SelectMatchingOpenCLType_t<s::vec<s::cl_int, 2>>,
                   s::vec<s::cl_int, 2>>::value,
      "");

  static_assert(
      std::is_same<d::SelectMatchingOpenCLType_t<s::multi_ptr<
                       s::cl_int, s::access::address_space::global_space>>,
                   s::multi_ptr<s::cl_int,
                                s::access::address_space::global_space>>::value,
      "");

  static_assert(
      std::is_same<
          d::SelectMatchingOpenCLType_t<s::multi_ptr<
              s::vec<s::cl_int, 2>, s::access::address_space::global_space>>,
          s::multi_ptr<s::vec<s::cl_int, 2>,
                       s::access::address_space::global_space>>::value,
      "");

  static_assert(std::is_same<d::SelectMatchingOpenCLType_t<s::longlong>,
                             s::cl_long>::value,
                "");

  static_assert(
      std::is_same<d::SelectMatchingOpenCLType_t<s::vec<s::longlong, 2>>,
                   s::vec<s::cl_long, 2>>::value,
      "");

  static_assert(
      std::is_same<d::SelectMatchingOpenCLType_t<s::multi_ptr<
                       s::longlong, s::access::address_space::global_space>>,
                   s::multi_ptr<s::cl_long,
                                s::access::address_space::global_space>>::value,
      "");

  static_assert(
      std::is_same<
          d::SelectMatchingOpenCLType_t<s::multi_ptr<
              s::vec<s::longlong, 2>, s::access::address_space::global_space>>,
          s::multi_ptr<s::vec<s::cl_long, 2>,
                       s::access::address_space::global_space>>::value,
      "");

  s::multi_ptr<int, s::access::address_space::global_space> mp;
  int *dp = mp;
}
