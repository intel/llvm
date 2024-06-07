// Header file with common utilities for testing SYCL 2020 image functionality.

#include <sycl/accessor_image.hpp>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/image.hpp>

using namespace sycl;

enum ImageType { Sampled, Unsampled };

template <int Dims>
using UnsampledCoordT =
    std::conditional_t<Dims == 1, int,
                       std::conditional_t<Dims == 2, int2, int4>>;

template <int Dims>
using SampledCoordT =
    std::conditional_t<Dims == 1, float,
                       std::conditional_t<Dims == 2, float2, float4>>;

template <ImageType ImgT, int Dims>
using CoordT = std::conditional_t<ImgT == ImageType::Sampled,
                                  SampledCoordT<Dims>, UnsampledCoordT<Dims>>;

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

template <addressing_mode AddrMode> std::string_view AddressingModeToString() {
  if constexpr (AddrMode == addressing_mode::clamp) {
    return "clamp";
  } else if constexpr (AddrMode == addressing_mode::clamp_to_edge) {
    return "clamp_to_edge";
  } else if constexpr (AddrMode == addressing_mode::repeat) {
    return "repeat";
  } else if constexpr (AddrMode == addressing_mode::mirrored_repeat) {
    return "mirrored_repeat";
  } else {
    return "none";
  }
}

template <image_format Format>
constexpr size_t BytesPerPixel =
    sizeof(typename FormatTraits<Format>::rep_elem_type) * 4;

template <image_format Format>
std::vector<typename FormatTraits<Format>::rep_elem_type>
GenerateData(size_t Size) {
  size_t TotalSize = Size * 4;
  std::vector<typename FormatTraits<Format>::rep_elem_type> Vec;
  Vec.reserve(TotalSize);
  for (size_t I = 0; I < TotalSize; ++I) {
    if constexpr (FormatTraits<Format>::Normalized) {
      Vec.push_back(static_cast<double>(I) / (TotalSize - 1));
    } else {
      Vec.push_back(I);
    }
  }
  return Vec;
}

template <int Dims>
std::vector<CoordT<ImageType::Sampled, Dims>> GetOffsetPermutations() {
  if constexpr (Dims == 1) {
    return {0.0, 0.25, 0.5, 0.75};
  } else if constexpr (Dims == 2) {
    std::vector<CoordT<ImageType::Sampled, Dims>> Perms;
    Perms.reserve(16);
    for (float OffsetX : {0.0, 0.25, 0.5, 0.75})
      for (float OffsetY : {0.0, 0.25, 0.5, 0.75})
        Perms.push_back({OffsetX, OffsetY});
    return Perms;
  } else {
    std::vector<CoordT<ImageType::Sampled, Dims>> Perms;
    Perms.reserve(64);
    for (float OffsetX : {0.0, 0.25, 0.5, 0.75})
      for (float OffsetY : {0.0, 0.25, 0.5, 0.75})
        for (float OffsetZ : {0.0, 0.25, 0.5, 0.75})
          Perms.push_back({OffsetX, OffsetY, OffsetZ, 0});
    return Perms;
  }
}

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

template <image_format Format, int Dims, typename ImageT>
range<2> getElementWisePitch(const ImageT &ImageRef) {
  range<2> OutRange{0, 0};
  if constexpr (Dims == 2) {
    range<1> Pitch = ImageRef.get_pitch();
    OutRange = range<2>{Pitch[0], 0};
  } else if constexpr (Dims == 3) {
    OutRange = ImageRef.get_pitch();
  }
  constexpr size_t ValueTSize =
      sizeof(typename FormatTraits<Format>::rep_elem_type) * 4;
  return {OutRange[0] / ValueTSize, OutRange[1] / ValueTSize};
}

template <ImageType ImgT, int Dims>
CoordT<ImgT, Dims> RangeToCoord(range<Dims> Range, int AdditionalElemVal = 1) {
  if constexpr (Dims == 1) {
    return static_cast<CoordT<ImgT, Dims>>(Range[0]);
  } else if constexpr (Dims == 2) {
    return CoordT<ImgT, Dims>{Range[0], Range[1]};
  } else {
    return CoordT<ImgT, Dims>{Range[0], Range[1], Range[2], AdditionalElemVal};
  }
}

template <ImageType ImgT, int Dims>
CoordT<ImgT, Dims> DelinearizeToCoord(size_t Idx, range<Dims> ImageRange,
                                      bool Normalize = false) {
  CoordT<ImgT, Dims> Out;
  if constexpr (Dims == 1) {
    Out = static_cast<CoordT<ImgT, Dims>>(Idx);
  } else if constexpr (Dims == 2) {
    Out = CoordT<ImgT, Dims>{Idx % ImageRange[0], Idx / ImageRange[0]};
  } else {
    Out = CoordT<ImgT, Dims>{Idx % ImageRange[0],
                             Idx / ImageRange[0] % ImageRange[1],
                             Idx / ImageRange[0] / ImageRange[1], 0};
  }
  if (Normalize)
    Out /= RangeToCoord<ImgT, Dims>(ImageRange, 2);
  return Out;
}

template <ImageType ImgT, int Dims>
size_t LinearizeCoord(CoordT<ImgT, Dims> Coords, range<2> ImagePitch) {
  if constexpr (Dims == 1) {
    return static_cast<size_t>(Coords);
  } else if constexpr (Dims == 2) {
    return static_cast<size_t>(Coords[0] + Coords[1] * ImagePitch[0]);
  } else {
    return static_cast<size_t>(Coords[0] + Coords[1] * ImagePitch[0] +
                               Coords[2] * ImagePitch[1]);
  }
}

template <int Dims, typename ExtraArgT = size_t>
CoordT<ImageType::Sampled, 3>
UpscaleCoord(CoordT<ImageType::Sampled, Dims> Coord, ExtraArgT ExtraArg = 0) {
  if constexpr (Dims == 1) {
    return {Coord, ExtraArg, ExtraArg, ExtraArg};
  } else if constexpr (Dims == 2) {
    return {Coord[0], Coord[1], ExtraArg, ExtraArg};
  } else {
    return Coord;
  }
}

template <int Dims>
CoordT<ImageType::Sampled, Dims>
DownscaleCoord(CoordT<ImageType::Sampled, 3> Coord) {
  if constexpr (Dims == 1) {
    return Coord[0];
  } else if constexpr (Dims == 2) {
    return {Coord[0], Coord[1]};
  } else {
    return Coord;
  }
}

template <ImageType ImgT, int Dims>
bool IsOutOfBounds(CoordT<ImgT, Dims> Coord, range<Dims> ImageRange) {
  if constexpr (Dims == 1) {
    return Coord < 0 || Coord > ImageRange[0] - 1;
  } else {
    for (size_t I = 0; I < Dims; ++I)
      if (Coord[I] < 0 || Coord[I] > ImageRange[I] - 1)
        return true;
    return false;
  }
}

template <typename T, int Dims> bool AllTrue(const vec<T, Dims> &Vec) {
  for (size_t I = 0; I < Dims; ++I)
    if (!Vec[I])
      return false;
  return true;
}

template <typename T, int Dims>
bool ApproxEq(const vec<T, Dims> &LHS, const vec<T, Dims> &RHS,
              T Precision = (T)0.1) {
  if constexpr (std::is_integral_v<T>)
    return AllTrue(sycl::abs(LHS - RHS) <= Precision);
  else
    return AllTrue(sycl::fabs(LHS - RHS) <= Precision);
}

template <typename T, int Dims>
std::ostream &operator<<(std::ostream &OS, const vec<T, Dims> &Vec) {
  OS << "{";
  for (size_t I = 0; I < Dims; ++I) {
    if (I)
      OS << ",";
    OS << Vec[I];
  }
  OS << "}";
  return OS;
}

template <image_format Format, ImageType ImgT, int Dims>
typename FormatTraits<Format>::pixel_type
SimulateRead(typename FormatTraits<Format>::rep_elem_type *RefData,
             CoordT<ImgT, Dims> Coord, range<2> ImagePitch,
             range<Dims> ImageRange, bool IsNormalized = false) {
  if (IsNormalized)
    Coord *= RangeToCoord<ImgT, Dims>(ImageRange);
  if (IsOutOfBounds<ImgT>(Coord, ImageRange))
    return typename FormatTraits<Format>::pixel_type{0};
  size_t I = 4 * LinearizeCoord<ImgT, Dims>(Coord, ImagePitch);
  return {RefData[I], RefData[I + 1], RefData[I + 2], RefData[I + 3]};
}

template <addressing_mode AddrMode, int Dims>
CoordT<ImageType::Sampled, Dims>
ApplyAddressingMode(CoordT<ImageType::Sampled, Dims> Coord,
                    range<Dims> ImageRange) {
  if constexpr (AddrMode == addressing_mode::none)
    return Coord;

  CoordT<ImageType::Sampled, Dims> ZeroCoord{0};
  CoordT<ImageType::Sampled, Dims> OneCoord{1};
  CoordT<ImageType::Sampled, Dims> RangeCoord =
      RangeToCoord<ImageType::Sampled, Dims>(ImageRange);

  if constexpr (AddrMode == addressing_mode::clamp) {
    return sycl::clamp(Coord, -OneCoord, RangeCoord);
  } else if constexpr (AddrMode == addressing_mode::clamp_to_edge ||
                       AddrMode == addressing_mode::mirrored_repeat) {
    return sycl::clamp(Coord, ZeroCoord, RangeCoord - OneCoord);
  } else if constexpr (AddrMode == addressing_mode::repeat) {
    if constexpr (Dims == 1) {
      return Coord + RangeCoord * (Coord < ZeroCoord) -
             RangeCoord * (Coord > (RangeCoord - OneCoord));
    } else {
      CoordT<ImageType::Sampled, Dims> NewCoord = Coord;
      for (int I = 0; I < Dims; ++I) {
        if (Coord[I] < 0)
          NewCoord[I] += RangeCoord[I];
        else if (Coord[I] > RangeCoord[I] - 1)
          NewCoord[I] -= RangeCoord[I];
      }
      return NewCoord;
    }
  }
}

template <image_format Format>
typename FormatTraits<Format>::pixel_type PickNewColor(size_t I,
                                                       size_t AccSize) {
  using RepElemT = typename FormatTraits<Format>::rep_elem_type;
  using PixelType = typename FormatTraits<Format>::pixel_type;

  size_t Idx = I * 4;

  // Pick a new color. Make sure it isn't too big for the data type.
  PixelType NewColor{Idx, Idx + 1, Idx + 2, Idx + 3};
  PixelType MaxPixelVal{std::numeric_limits<RepElemT>::max()};
  NewColor = sycl::min(NewColor, MaxPixelVal);
  if constexpr (FormatTraits<Format>::Normalized)
    NewColor /= AccSize * 4;
  return NewColor;
}

// Implemented as specified by the OpenCL 1.2 specification for
// CLK_FILTER_NEAREST.
template <image_format Format, addressing_mode AddrMode, int Dims>
typename FormatTraits<Format>::pixel_type
ReadNearest(typename FormatTraits<Format>::rep_elem_type *RefData,
            CoordT<ImageType::Sampled, Dims> Coord, range<2> ImagePitch,
            range<Dims> ImageRange, bool Normalized) {
  CoordT<ImageType::Sampled, Dims> AdjCoord = Coord;
  if constexpr (AddrMode == addressing_mode::repeat) {
    assert(Normalized);
    AdjCoord -= sycl::floor(AdjCoord);
    AdjCoord *= RangeToCoord<ImageType::Sampled, Dims>(ImageRange);
    AdjCoord = sycl::floor(AdjCoord);
  } else if constexpr (AddrMode == addressing_mode::mirrored_repeat) {
    assert(Normalized);
    AdjCoord = 2.0f * sycl::rint(0.5f * Coord);
    AdjCoord = sycl::fabs(Coord - AdjCoord);
    AdjCoord *= RangeToCoord<ImageType::Sampled, Dims>(ImageRange);
    AdjCoord = sycl::floor(AdjCoord);
  } else {
    if (Normalized)
      AdjCoord *= RangeToCoord<ImageType::Sampled, Dims>(ImageRange);
    AdjCoord = sycl::floor(AdjCoord);
  }
  AdjCoord = ApplyAddressingMode<AddrMode>(AdjCoord, ImageRange);
  return SimulateRead<Format, ImageType::Sampled>(RefData, AdjCoord, ImagePitch,
                                                  ImageRange, false);
}

// Implemented as specified by the OpenCL 1.2 specification for
// CLK_FILTER_LINEAR.
template <image_format Format, addressing_mode AddrMode, int Dims>
float4 CalcLinearRead(typename FormatTraits<Format>::rep_elem_type *RefData,
                      CoordT<ImageType::Sampled, Dims> Coord,
                      range<2> ImagePitch, range<Dims> ImageRange,
                      bool Normalized) {
  using UpscaledCoordT = CoordT<ImageType::Sampled, 3>;

  auto Read = [&](UpscaledCoordT UpCoord) {
    auto DownCoord = DownscaleCoord<Dims>(UpCoord);
    return SimulateRead<Format, ImageType::Sampled>(
        RefData, DownCoord, ImagePitch, ImageRange, false);
  };

  CoordT<ImageType::Sampled, Dims> AdjCoord = Coord;
  if constexpr (AddrMode == addressing_mode::repeat) {
    assert(Normalized);
    AdjCoord -= sycl::floor(AdjCoord);
    AdjCoord *= RangeToCoord<ImageType::Sampled, Dims>(ImageRange);
  } else if constexpr (AddrMode == addressing_mode::mirrored_repeat) {
    assert(Normalized);
    AdjCoord = 2.0f * sycl::rint(0.5f * Coord);
    AdjCoord = sycl::fabs(Coord - AdjCoord);
    AdjCoord *= RangeToCoord<ImageType::Sampled, Dims>(ImageRange);
  } else {
    if (Normalized)
      AdjCoord *= RangeToCoord<ImageType::Sampled, Dims>(ImageRange);
  }

  auto Prev = sycl::floor(AdjCoord - 0.5f);
  auto Next = Prev + 1;
  auto CA = (AdjCoord - 0.5f) - Prev;

  Prev = ApplyAddressingMode<AddrMode>(Prev, ImageRange);
  Next = ApplyAddressingMode<AddrMode>(Next, ImageRange);

  auto UPrev = UpscaleCoord<Dims>(Prev);
  auto UNext = UpscaleCoord<Dims>(Next);
  auto UCA = UpscaleCoord<Dims>(CA, 1);

  auto CA000 = Read(UpscaledCoordT{UPrev[0], UPrev[1], UPrev[2], 0})
                   .template convert<float>() *
               (1 - UCA[0]) * (1 - UCA[1]) * (1 - UCA[2]);
  auto CA100 = Read(UpscaledCoordT{UNext[0], UPrev[1], UPrev[2], 0})
                   .template convert<float>() *
               UCA[0] * (1 - UCA[1]) * (1 - UCA[2]);
  auto CA010 = Read(UpscaledCoordT{UPrev[0], UNext[1], UPrev[2], 0})
                   .template convert<float>() *
               (1 - UCA[0]) * UCA[1] * (1 - UCA[2]);
  auto CA110 = Read(UpscaledCoordT{UNext[0], UNext[1], UPrev[2], 0})
                   .template convert<float>() *
               UCA[0] * UCA[1] * (1 - UCA[2]);
  auto CA001 = Read(UpscaledCoordT{UPrev[0], UPrev[1], UNext[2], 0})
                   .template convert<float>() *
               (1 - UCA[0]) * (1 - UCA[1]) * UCA[2];
  auto CA101 = Read(UpscaledCoordT{UNext[0], UPrev[1], UNext[2], 0})
                   .template convert<float>() *
               UCA[0] * (1 - UCA[1]) * UCA[2];
  auto CA011 = Read(UpscaledCoordT{UPrev[0], UNext[1], UNext[2], 0})
                   .template convert<float>() *
               (1 - UCA[0]) * UCA[1] * UCA[2];
  auto CA111 = Read(UpscaledCoordT{UNext[0], UNext[1], UNext[2], 0})
                   .template convert<float>() *
               UCA[0] * UCA[1] * UCA[2];
  return CA000 + CA100 + CA010 + CA110 + CA001 + CA101 + CA011 + CA111;
}
