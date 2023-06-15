// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>
#include <variant>

struct ACopyable {
  int i;
  ACopyable() = default;
  ACopyable(int _i) : i(_i) {}
  ACopyable(const ACopyable &x) : i(x.i) {}
};

template <> struct sycl::is_device_copyable<ACopyable> : std::true_type {};

int main() {
  static_assert(sycl::is_device_copyable_v<std::pair<int, float>>);
  static_assert(sycl::is_device_copyable_v<std::pair<ACopyable, float>>);
  static_assert(sycl::is_device_copyable_v<std::tuple<int, float, bool>>);
  static_assert(sycl::is_device_copyable_v<std::tuple<ACopyable, float, bool>>);
  static_assert(sycl::is_device_copyable_v<std::variant<int, float, bool>>);
  static_assert(sycl::is_device_copyable_v<std::variant<ACopyable>>);
  static_assert(sycl::is_device_copyable_v<std::array<int, 513>>);
  static_assert(sycl::is_device_copyable_v<std::array<ACopyable, 513>>);
  static_assert(sycl::is_device_copyable_v<std::optional<int>>);
  static_assert(sycl::is_device_copyable_v<std::optional<ACopyable>>);
  static_assert(sycl::is_device_copyable_v<std::string_view>);
#if __cpp_lib_span >= 202002
  static_assert(sycl::is_device_copyable_v<std::span<int>>);
#endif
  static_assert(sycl::is_device_copyable_v<const sycl::span<int>>);

  // sycl::vec
  static_assert(sycl::is_device_copyable_v<sycl::vec<int8_t, 1>>);
  static_assert(sycl::is_device_copyable_v<sycl::vec<uint8_t, 1>>);
  static_assert(sycl::is_device_copyable_v<sycl::vec<int16_t, 1>>);
  static_assert(sycl::is_device_copyable_v<sycl::vec<uint16_t, 1>>);
  static_assert(sycl::is_device_copyable_v<sycl::vec<int32_t, 1>>);
  static_assert(sycl::is_device_copyable_v<sycl::vec<uint32_t, 1>>);
  static_assert(sycl::is_device_copyable_v<sycl::vec<int64_t, 1>>);
  static_assert(sycl::is_device_copyable_v<sycl::vec<uint64_t, 1>>);
  static_assert(sycl::is_device_copyable_v<sycl::vec<sycl::half, 1>>);
  static_assert(sycl::is_device_copyable_v<sycl::vec<float, 1>>);
  static_assert(sycl::is_device_copyable_v<sycl::vec<double, 1>>);
  static_assert(sycl::is_device_copyable_v<sycl::char2>);
  static_assert(sycl::is_device_copyable_v<sycl::uchar2>);
  static_assert(sycl::is_device_copyable_v<sycl::short2>);
  static_assert(sycl::is_device_copyable_v<sycl::ushort2>);
  static_assert(sycl::is_device_copyable_v<sycl::int2>);
  static_assert(sycl::is_device_copyable_v<sycl::uint2>);
  static_assert(sycl::is_device_copyable_v<sycl::long2>);
  static_assert(sycl::is_device_copyable_v<sycl::ulong2>);
  static_assert(sycl::is_device_copyable_v<sycl::half2>);
  static_assert(sycl::is_device_copyable_v<sycl::float2>);
  static_assert(sycl::is_device_copyable_v<sycl::double2>);
  static_assert(sycl::is_device_copyable_v<sycl::char3>);
  static_assert(sycl::is_device_copyable_v<sycl::uchar3>);
  static_assert(sycl::is_device_copyable_v<sycl::short3>);
  static_assert(sycl::is_device_copyable_v<sycl::ushort3>);
  static_assert(sycl::is_device_copyable_v<sycl::int3>);
  static_assert(sycl::is_device_copyable_v<sycl::uint3>);
  static_assert(sycl::is_device_copyable_v<sycl::long3>);
  static_assert(sycl::is_device_copyable_v<sycl::ulong3>);
  static_assert(sycl::is_device_copyable_v<sycl::half3>);
  static_assert(sycl::is_device_copyable_v<sycl::float3>);
  static_assert(sycl::is_device_copyable_v<sycl::double3>);
  static_assert(sycl::is_device_copyable_v<sycl::char4>);
  static_assert(sycl::is_device_copyable_v<sycl::uchar4>);
  static_assert(sycl::is_device_copyable_v<sycl::short4>);
  static_assert(sycl::is_device_copyable_v<sycl::ushort4>);
  static_assert(sycl::is_device_copyable_v<sycl::int4>);
  static_assert(sycl::is_device_copyable_v<sycl::uint4>);
  static_assert(sycl::is_device_copyable_v<sycl::long4>);
  static_assert(sycl::is_device_copyable_v<sycl::ulong4>);
  static_assert(sycl::is_device_copyable_v<sycl::half4>);
  static_assert(sycl::is_device_copyable_v<sycl::float4>);
  static_assert(sycl::is_device_copyable_v<sycl::double4>);
  static_assert(sycl::is_device_copyable_v<sycl::char8>);
  static_assert(sycl::is_device_copyable_v<sycl::uchar8>);
  static_assert(sycl::is_device_copyable_v<sycl::short8>);
  static_assert(sycl::is_device_copyable_v<sycl::ushort8>);
  static_assert(sycl::is_device_copyable_v<sycl::int8>);
  static_assert(sycl::is_device_copyable_v<sycl::uint8>);
  static_assert(sycl::is_device_copyable_v<sycl::long8>);
  static_assert(sycl::is_device_copyable_v<sycl::ulong8>);
  static_assert(sycl::is_device_copyable_v<sycl::half8>);
  static_assert(sycl::is_device_copyable_v<sycl::float8>);
  static_assert(sycl::is_device_copyable_v<sycl::double8>);
  static_assert(sycl::is_device_copyable_v<sycl::char16>);
  static_assert(sycl::is_device_copyable_v<sycl::uchar16>);
  static_assert(sycl::is_device_copyable_v<sycl::short16>);
  static_assert(sycl::is_device_copyable_v<sycl::ushort16>);
  static_assert(sycl::is_device_copyable_v<sycl::int16>);
  static_assert(sycl::is_device_copyable_v<sycl::uint16>);
  static_assert(sycl::is_device_copyable_v<sycl::long16>);
  static_assert(sycl::is_device_copyable_v<sycl::ulong16>);
  static_assert(sycl::is_device_copyable_v<sycl::half16>);
  static_assert(sycl::is_device_copyable_v<sycl::float16>);
  static_assert(sycl::is_device_copyable_v<sycl::double16>);

  // const
  static_assert(sycl::is_device_copyable_v<const std::pair<int, float>>);
  static_assert(sycl::is_device_copyable_v<const std::pair<ACopyable, float>>);
  static_assert(sycl::is_device_copyable_v<const std::tuple<int, float, bool>>);
  static_assert(
      sycl::is_device_copyable_v<const std::tuple<ACopyable, float, bool>>);
  static_assert(
      sycl::is_device_copyable_v<const std::variant<int, float, bool>>);
  static_assert(sycl::is_device_copyable_v<const std::variant<ACopyable>>);
  static_assert(sycl::is_device_copyable_v<const std::array<int, 513>>);
  static_assert(sycl::is_device_copyable_v<const std::array<ACopyable, 513>>);
  static_assert(sycl::is_device_copyable_v<const std::optional<int>>);
  static_assert(sycl::is_device_copyable_v<const std::optional<ACopyable>>);
  static_assert(sycl::is_device_copyable_v<const std::string_view>);
#if __cpp_lib_span >= 202002
  static_assert(sycl::is_device_copyable_v<const std::span<int>>);
#endif
  static_assert(sycl::is_device_copyable_v<const sycl::span<int>>);

  // volatile
  static_assert(sycl::is_device_copyable_v<volatile std::pair<int, float>>);
  static_assert(
      sycl::is_device_copyable_v<volatile std::pair<ACopyable, float>>);
  static_assert(
      sycl::is_device_copyable_v<volatile std::tuple<int, float, bool>>);
  static_assert(
      sycl::is_device_copyable_v<volatile std::tuple<ACopyable, float, bool>>);
  static_assert(
      sycl::is_device_copyable_v<volatile std::variant<int, float, bool>>);
  static_assert(sycl::is_device_copyable_v<volatile std::variant<ACopyable>>);
  static_assert(sycl::is_device_copyable_v<volatile std::array<int, 513>>);
  static_assert(
      sycl::is_device_copyable_v<volatile std::array<ACopyable, 513>>);
  static_assert(sycl::is_device_copyable_v<volatile std::optional<int>>);
  static_assert(sycl::is_device_copyable_v<volatile std::optional<ACopyable>>);
  static_assert(sycl::is_device_copyable_v<volatile std::string_view>);
#if __cpp_lib_span >= 202002
  static_assert(sycl::is_device_copyable_v<volatile std::span<int>>);
#endif
  static_assert(sycl::is_device_copyable_v<volatile sycl::span<int>>);

  // const volatile
  static_assert(
      sycl::is_device_copyable_v<const volatile std::pair<int, float>>);
  static_assert(
      sycl::is_device_copyable_v<const volatile std::pair<ACopyable, float>>);
  static_assert(
      sycl::is_device_copyable_v<const volatile std::tuple<int, float, bool>>);
  static_assert(sycl::is_device_copyable_v<
                const volatile std::tuple<ACopyable, float, bool>>);
  static_assert(sycl::is_device_copyable_v<
                const volatile std::variant<int, float, bool>>);
  static_assert(
      sycl::is_device_copyable_v<const volatile std::variant<ACopyable>>);
  static_assert(
      sycl::is_device_copyable_v<const volatile std::array<int, 513>>);
  static_assert(
      sycl::is_device_copyable_v<const volatile std::array<ACopyable, 513>>);
  static_assert(sycl::is_device_copyable_v<const volatile std::optional<int>>);
  static_assert(
      sycl::is_device_copyable_v<const volatile std::optional<ACopyable>>);
  static_assert(sycl::is_device_copyable_v<const volatile std::string_view>);
#if __cpp_lib_span >= 202002
  static_assert(sycl::is_device_copyable_v<const volatile std::span<int>>);
#endif
  static_assert(sycl::is_device_copyable_v<const volatile sycl::span<int>>);

  return 0;
}
