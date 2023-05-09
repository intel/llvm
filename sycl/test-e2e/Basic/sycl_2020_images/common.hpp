// Header file with common utilities for testing SYCL 2020 image functionality.

#include <sycl/sycl.hpp>

using namespace sycl;

template <image_format Format> struct FormatTraits;
template <> struct FormatTraits<image_format::r8g8b8a8_unorm> {
  using pixel_type = float4;
  using rep_elem_type = int8_t;
  static constexpr bool Normalized = true;
  static constexpr std::string_view Name = "r8g8b8a8_unorm";
};
template <> struct FormatTraits<image_format::r16g16b16a16_unorm> {
  using pixel_type = float4;
  using rep_elem_type = int16_t;
  static constexpr bool Normalized = true;
  static constexpr std::string_view Name = "r16g16b16a16_unorm";
};
template <> struct FormatTraits<image_format::r8g8b8a8_sint> {
  using pixel_type = int4;
  using rep_elem_type = int8_t;
  static constexpr bool Normalized = false;
  static constexpr std::string_view Name = "r8g8b8a8_sint";
};
template <> struct FormatTraits<image_format::r16g16b16a16_sint> {
  using pixel_type = int4;
  using rep_elem_type = int16_t;
  static constexpr bool Normalized = false;
  static constexpr std::string_view Name = "r16g16b16a16_sint";
};
template <> struct FormatTraits<image_format::r32b32g32a32_sint> {
  using pixel_type = int4;
  using rep_elem_type = int32_t;
  static constexpr bool Normalized = false;
  static constexpr std::string_view Name = "r32b32g32a32_sint";
};
template <> struct FormatTraits<image_format::r8g8b8a8_uint> {
  using pixel_type = uint4;
  using rep_elem_type = uint8_t;
  static constexpr bool Normalized = false;
  static constexpr std::string_view Name = "r8g8b8a8_uint";
};
template <> struct FormatTraits<image_format::r16g16b16a16_uint> {
  using pixel_type = uint4;
  using rep_elem_type = uint16_t;
  static constexpr bool Normalized = false;
  static constexpr std::string_view Name = "r16g16b16a16_uint";
};
template <> struct FormatTraits<image_format::r32b32g32a32_uint> {
  using pixel_type = uint4;
  using rep_elem_type = uint32_t;
  static constexpr bool Normalized = false;
  static constexpr std::string_view Name = "r32b32g32a32_uint";
};
template <> struct FormatTraits<image_format::r16b16g16a16_sfloat> {
  using pixel_type = half4;
  using rep_elem_type = half;
  static constexpr bool Normalized = false;
  static constexpr std::string_view Name = "r16b16g16a16_sfloat";
};
template <> struct FormatTraits<image_format::r32g32b32a32_sfloat> {
  using pixel_type = float4;
  using rep_elem_type = float;
  static constexpr bool Normalized = false;
  static constexpr std::string_view Name = "r32g32b32a32_sfloat";
};
template <> struct FormatTraits<image_format::b8g8r8a8_unorm> {
  using pixel_type = float4;
  using rep_elem_type = int8_t;
  static constexpr bool Normalized = true;
  static constexpr std::string_view Name = "b8g8r8a8_unorm";
};

template <image_format Format>
constexpr size_t BytesPerPixel =
    sizeof(typename FormatTraits<Format>::rep_elem_type) * 4;

template <int Dims> range<Dims> CreateImageRange(size_t X, size_t Y, size_t Z) {
  if constexpr (Dims == 1) {
    return range<Dims>(X);
  } else if constexpr (Dims == 2) {
    return range<Dims>(X, Y);
  } else {
    return range<Dims>(X, Y, Z);
  }
}

template <int Dims>
std::ostream &operator<<(std::ostream &OS, const range<Dims> &Range) {
  OS << "<";
  for (size_t I = 0; I < Dims; ++I) {
    if (I)
      OS << ",";
    OS << Range[I];
  }
  OS << ">";
  return OS;
}
