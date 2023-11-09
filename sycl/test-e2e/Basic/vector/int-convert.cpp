// Basic acceptance test which checks vec::convert implementation on both
// host and device. Coverage is limited to vec<T, 4> only, rest of vector sizes
// are covered by SYCL-CTS.
//
// Macro is passed to silence warnings about sycl::byte
//
// RUN: %{build} -o %t.out -DSYCL2020_DISABLE_DEPRECATION_WARNINGS
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

// Debug prints are hidden under macro to reduce amount of output in CI runs
// and thus speed up tests. However, they are useful when debugging the test
// locally and can be quickly turned on in there.
#ifdef ENABLE_DEBUG_OUTPUT

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

#define DEBUG_PRINT(x) std::cout << x << std::endl;

#else
#define DEBUG_PRINT(x)
#endif

template <typename T>
void check_vectors_equal(sycl::vec<T, 4> a, sycl::vec<T, 4> b) {
  bool all_good =
      a.x() == b.x() && a.y() == b.y() && a.z() == b.z() && a.w() == b.w();
  if (!all_good) {
    DEBUG_PRINT("host and device results mismatch:");
    DEBUG_PRINT(
        "\t{" << static_cast<int>(a.x()) << ", " << static_cast<int>(a.y())
              << ", " << static_cast<int>(a.z()) << ", "
              << static_cast<int>(a.w()) << "} vs {" << static_cast<int>(b.x())
              << ", " << static_cast<int>(b.y()) << ", "
              << static_cast<int>(b.z()) << ", " << static_cast<int>(b.w())
              << "}");
  }
  assert(all_good);
}

template <typename From, typename To> void check_convert() {
  DEBUG_PRINT("checking vec<" << to_string<From>() << ", 4>::convert<"
                              << to_string<To>() << ">()");

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

  // Host and device results must match.
  check_vectors_equal(hostResult, deviceResult);

  // And they should match with a reference, which is for integer conversions
  // can be computed with a simple static_cast.
  // Strictly speaking, integer conversions are underspecified in the SYCL 2020
  // spec, but `static_cast` implementation matches SYCL-CTS, so we will leave
  // it here for now as well.
  // See https://github.com/KhronosGroup/SYCL-Docs/issues/492
  assert(deviceResult.x() == static_cast<To>(input.x()));
  assert(deviceResult.y() == static_cast<To>(input.y()));
  assert(deviceResult.z() == static_cast<To>(input.z()));
  assert(deviceResult.w() == static_cast<To>(input.w()));
}

template <class T>
constexpr auto has_unsigned_v =
    std::is_integral_v<T> && !std::is_same_v<T, bool> &&
    !std::is_same_v<T, sycl::byte> && !std::is_same_v<T, std::byte>;

template <typename From, typename To> void check_signed_unsigned_convert_to() {
  check_convert<From, To>();
  if constexpr (has_unsigned_v<To>)
    check_convert<From, std::make_unsigned_t<To>>();
  if constexpr (has_unsigned_v<From>)
    check_convert<std::make_unsigned_t<From>, To>();
  if constexpr (has_unsigned_v<To> && has_unsigned_v<From>)
    check_convert<std::make_unsigned_t<From>, std::make_unsigned_t<To>>();
}

template <typename From> void check_convert_from() {
  check_signed_unsigned_convert_to<From, sycl::byte>();
  // FIXME: enable test cases below once compilation issues for them are fixed
  // check_signed_unsigned_convert_to<From, std::byte>();
  check_signed_unsigned_convert_to<From, std::int8_t>();
  check_signed_unsigned_convert_to<From, std::int16_t>();
  check_signed_unsigned_convert_to<From, std::int32_t>();
  check_signed_unsigned_convert_to<From, std::int64_t>();
  check_signed_unsigned_convert_to<From, bool>();
  check_signed_unsigned_convert_to<From, char>();
  check_signed_unsigned_convert_to<From, signed char>();
  check_signed_unsigned_convert_to<From, short>();
  check_signed_unsigned_convert_to<From, int>();
  check_signed_unsigned_convert_to<From, long>();
  check_signed_unsigned_convert_to<From, long long>();
}

int main() {
  check_convert_from<sycl::byte>();
  // FIXME: enable test cases below once compilation issues for them are fixed
  // check_convert_from<std::byte>();
  check_convert_from<std::int8_t>();
  check_convert_from<std::int16_t>();
  check_convert_from<std::int32_t>();
  check_convert_from<std::int64_t>();
  check_convert_from<char>();
  check_convert_from<signed char>();
  check_convert_from<short>();
  check_convert_from<int>();
  check_convert_from<long>();
  check_convert_from<long long>();
  check_convert_from<bool>();
}
