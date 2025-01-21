// RUN: %clangxx -fsycl -fsyntax-only %s
//
// This test aims to check that the implementation provides aliases for
// interoperability with OpenCL in accordance with section C.7.1 Data types of
// SYCL 2020 specification, revision 6

#include <sycl/sycl.hpp>

#include <type_traits>

int main() {
  // cl_bool
  // Alias to a conditional data type which can be either true or false.
  // The value true expands to the integer constant 1 and the value false
  // expands to the integer constant 0.
  constexpr sycl::opencl::cl_bool True = true;
  constexpr sycl::opencl::cl_bool False = false;
  static_assert((int)True == 1);
  static_assert((int)False == 0);

  // cl_char
  // Alias to a signed 1-bit integer, as defined by the C++ core language.
  static_assert(sizeof(sycl::opencl::cl_char) == 1);
  static_assert(std::is_signed_v<sycl::opencl::cl_char>);
  static_assert(std::is_integral_v<sycl::opencl::cl_char>);

  // cl_uchar
  // Alias to an unsigned 1-bit integer, as defined by the C++ core language.
  static_assert(sizeof(sycl::opencl::cl_uchar) == 1);
  static_assert(std::is_unsigned_v<sycl::opencl::cl_uchar>);
  static_assert(std::is_integral_v<sycl::opencl::cl_uchar>);

  // cl_short
  // Alias to a signed 2-bit integer, as defined by the C++ core language.
  static_assert(sizeof(sycl::opencl::cl_short) == 2);
  static_assert(std::is_signed_v<sycl::opencl::cl_short>);
  static_assert(std::is_integral_v<sycl::opencl::cl_short>);

  // cl_ushort
  // Alias to an unsigned 2-bit integer, as defined by the C++ core language.
  static_assert(sizeof(sycl::opencl::cl_ushort) == 2);
  static_assert(std::is_unsigned_v<sycl::opencl::cl_ushort>);
  static_assert(std::is_integral_v<sycl::opencl::cl_ushort>);

  // cl_int
  // Alias to a signed 4-bit integer, as defined by the C++ core language.
  static_assert(sizeof(sycl::opencl::cl_int) == 4);
  static_assert(std::is_signed_v<sycl::opencl::cl_int>);
  static_assert(std::is_integral_v<sycl::opencl::cl_int>);

  // cl_uint
  // Alias to an unsigned 4-bit integer, as defined by the C++ core language.
  static_assert(sizeof(sycl::opencl::cl_uint) == 4);
  static_assert(std::is_unsigned_v<sycl::opencl::cl_uint>);
  static_assert(std::is_integral_v<sycl::opencl::cl_uint>);

  // cl_long
  // Alias to a signed 8-bit integer, as defined by the C++ core language.
  static_assert(sizeof(sycl::opencl::cl_long) == 8);
  static_assert(std::is_signed_v<sycl::opencl::cl_long>);
  static_assert(std::is_integral_v<sycl::opencl::cl_long>);

  // cl_ulong
  // Alias to an unsigned 8-bit integer, as defined by the C++ core language.
  static_assert(sizeof(sycl::opencl::cl_ulong) == 8);
  static_assert(std::is_unsigned_v<sycl::opencl::cl_ulong>);
  static_assert(std::is_integral_v<sycl::opencl::cl_ulong>);

  // TODO: how can we check that the type conforms to IEEE 754?
  // There is std::numeric_limits<T>::is_iec559 - is it enough?

  // cl_float
  // Alias to a 4-bit floating-point. The float data type must conform to the
  // IEEE 754 single precision storage format.
  static_assert(sizeof(sycl::opencl::cl_float) == 4);
  static_assert(std::is_floating_point_v<sycl::opencl::cl_float>);

  // cl_double
  // Alias to a 8-bit floating-point. The double data type must conform to the
  // IEEE 754 double precision storage format.
  static_assert(sizeof(sycl::opencl::cl_double) == 8);
  static_assert(std::is_floating_point_v<sycl::opencl::cl_double>);

  // cl_half
  // Alias to a 2-bit floating-point. The half data type must conform to the
  // IEEE 754-2001 half precision storage format.
  static_assert(sizeof(sycl::opencl::cl_half) == 2);
  // TODO: do we need std::is_floating_point to return true for
  // sycl::opencl::cl_half?
  // We alias it to our custom class sycl::half and C++ spec says it is UB
  // to provide specializations for std::is_floating_point type trait
  // static_assert(std::is_floating_point_v<sycl::opencl::cl_half>);

  return 0;
}
