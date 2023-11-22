// Basic acceptance test which checks vec::convert implementation on both
// host and device. Coverage is limited to vec<T, 4> only, rest of vector sizes
// are covered by SYCL-CTS.
//
// Macro is passed to silence warnings about sycl::byte
//
// XFAIL: cuda
// FIXME: un-xfail the test once intel/llvm#11840 is resolved
//
// RUN: %{build} -o %t.out -DSYCL2020_DISABLE_DEPRECATION_WARNINGS
// RUN: %{run} %t.out
//
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -DSYCL2020_DISABLE_DEPRECATION_WARNINGS %s -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{  %{run} %t2.out %}

#include <sycl/sycl.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

template <typename T> std::string to_string() { return "unknown type"; }
template <> std::string to_string<std::byte>() { return "std::byte"; }
template <> std::string to_string<char>() { return "char"; }
template <> std::string to_string<signed char>() { return "signed char"; }
template <> std::string to_string<short>() { return "short"; }
template <> std::string to_string<int>() { return "int"; }
template <> std::string to_string<long>() { return "long"; }
template <> std::string to_string<long long>() { return "long long"; }
template <> std::string to_string<unsigned char>() { return "unsigned char"; }
template <> std::string to_string<unsigned short>() { return "unsigned short"; }
template <> std::string to_string<unsigned int>() { return "unsigned int"; }
template <> std::string to_string<unsigned long>() { return "unsigned long"; }
template <> std::string to_string<unsigned long long>() {
  return "unsigned long long";
}
template <> std::string to_string<bool>() { return "bool"; }

template <typename T>
bool check_vectors_equal(sycl::vec<T, 4> a, sycl::vec<T, 4> b,
                         const std::string &fail_message) {
  bool result =
      a.x() == b.x() && a.y() == b.y() && a.z() == b.z() && a.w() == b.w();
  if (!result) {
    std::cout << fail_message << std::endl;
    std::cout << "\t{" << static_cast<int>(a.x()) << ", "
              << static_cast<int>(a.y()) << ", " << static_cast<int>(a.z())
              << ", " << static_cast<int>(a.w()) << "} vs {"
              << static_cast<int>(b.x()) << ", " << static_cast<int>(b.y())
              << ", " << static_cast<int>(b.z()) << ", "
              << static_cast<int>(b.w()) << "}" << std::endl;
  }

  return result;
}

template <typename From, typename To> bool check_convert() {
  sycl::vec<From, 4> input;
  if constexpr (std::is_signed_v<From>) {
    input = sycl::vec<From, 4>{static_cast<From>(37), static_cast<From>(0),
                               static_cast<From>(-11), static_cast<From>(13)};
  } else {
    input = sycl::vec<From, 4>{static_cast<From>(37), static_cast<From>(0),
                               static_cast<From>(11), static_cast<From>(13)};
  }

  sycl::vec<To, 4> hostResult = input.template convert<To>();

  sycl::buffer<sycl::vec<To, 4>> buf(sycl::range{1});
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
     sycl::accessor acc(buf, cgh);
     cgh.single_task([=]() { acc[0] = input.template convert<To>(); });
   }).wait();

  auto acc = buf.get_host_access();
  auto deviceResult = acc[0];

  std::string test =
      "(vec<" + to_string<From>() + ", 4>::convert<" + to_string<To>() + ">)";

  // Host and device results must match.
  bool host_and_device_match = check_vectors_equal(
      hostResult, deviceResult, "host and device results do not match " + test);
  // And they should match with a reference, which is for integer conversions
  // can be computed with a simple static_cast.
  // Strictly speaking, integer conversions are underspecified in the SYCL 2020
  // spec, but `static_cast` implementation matches SYCL-CTS, so we will leave
  // it here for now as well.
  // See https://github.com/KhronosGroup/SYCL-Docs/issues/492
  sycl::vec<To, 4> reference{
      static_cast<To>(input.x()), static_cast<To>(input.y()),
      static_cast<To>(input.z()), static_cast<To>(input.w())};
  bool device_matches_reference = check_vectors_equal(
      deviceResult, reference, "device results don't match reference " + test);
  bool host_matches_reference = check_vectors_equal(
      hostResult, reference, "host resutls don't match reference " + test);

  return host_and_device_match && device_matches_reference &&
         host_matches_reference;
}

template <class T>
constexpr auto has_unsigned_v =
    std::is_integral_v<T> && !std::is_same_v<T, bool> &&
    !std::is_same_v<T, sycl::byte> && !std::is_same_v<T, std::byte>;

template <typename From, typename To> bool check_signed_unsigned_convert_to() {
  bool pass = true;
  pass &= check_convert<From, To>();
  if constexpr (has_unsigned_v<To>)
    pass &= check_convert<From, std::make_unsigned_t<To>>();
  if constexpr (has_unsigned_v<From>)
    pass &= check_convert<std::make_unsigned_t<From>, To>();
  if constexpr (has_unsigned_v<To> && has_unsigned_v<From>)
    pass &=
        check_convert<std::make_unsigned_t<From>, std::make_unsigned_t<To>>();
  return pass;
}

template <typename From> bool check_convert_from() {
  bool pass = true;
  pass &= check_signed_unsigned_convert_to<From, sycl::byte>();
  pass &= check_signed_unsigned_convert_to<From, std::byte>();
  pass &= check_signed_unsigned_convert_to<From, std::int8_t>();
  pass &= check_signed_unsigned_convert_to<From, std::int16_t>();
  pass &= check_signed_unsigned_convert_to<From, std::int32_t>();
  pass &= check_signed_unsigned_convert_to<From, std::int64_t>();
  pass &= check_signed_unsigned_convert_to<From, bool>();
  pass &= check_signed_unsigned_convert_to<From, char>();
  pass &= check_signed_unsigned_convert_to<From, signed char>();
  pass &= check_signed_unsigned_convert_to<From, short>();
  pass &= check_signed_unsigned_convert_to<From, int>();
  pass &= check_signed_unsigned_convert_to<From, long>();
  pass &= check_signed_unsigned_convert_to<From, long long>();

  return pass;
}

int main() {
  bool pass = true;
  pass &= check_convert_from<sycl::byte>();
  pass &= check_convert_from<std::byte>();
  pass &= check_convert_from<std::int8_t>();
  pass &= check_convert_from<std::int16_t>();
  pass &= check_convert_from<std::int32_t>();
  pass &= check_convert_from<std::int64_t>();
  pass &= check_convert_from<char>();
  pass &= check_convert_from<signed char>();
  pass &= check_convert_from<short>();
  pass &= check_convert_from<int>();
  pass &= check_convert_from<long>();
  pass &= check_convert_from<long long>();
  pass &= check_convert_from<bool>();

  return static_cast<int>(!pass);
}
