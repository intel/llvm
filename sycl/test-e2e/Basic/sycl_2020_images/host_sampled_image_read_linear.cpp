// RUN: %{build} -o %t.out
// TODO: Consider moving to sycl/test as this is device-independent.
// RUN: %{run-unfiltered-devices} %t.out

// Tests for indirect read of sampled_image using host accessors and linear
// filtering mode.

#include "common.hpp"

constexpr size_t IMAGE_WIDTH = 5;
constexpr size_t IMAGE_HEIGHT = 4;
constexpr size_t IMAGE_DEPTH = 2;

constexpr size_t IMAGE_PITCH_WIDTH = 7;
constexpr size_t IMAGE_PITCH_HEIGHT = 5 * IMAGE_PITCH_WIDTH;

template <image_format Format, addressing_mode AddrMode,
          coordinate_normalization_mode CoordNormMode, int Dims>
bool checkSampledImageHostReadLinear(
    sampled_image<Dims> &Image,
    typename FormatTraits<Format>::rep_elem_type *RefData) {
  using PixelType = typename FormatTraits<Format>::pixel_type;
  constexpr ImageType ImgType = ImageType::Sampled;
  constexpr bool Normalized =
      CoordNormMode == coordinate_normalization_mode::normalized;

  host_sampled_image_accessor<PixelType, Dims> Acc(Image);
  assert(Image.size() == Acc.size());

  range<Dims> ImageRange = Image.get_range();
  range<2> ImagePitch = getElementWisePitch<Format, Dims>(Image);

  // Get the offset permutations to add to the direct coordinates to check
  // non-direct reads.
  auto Offsets = GetOffsetPermutations<Dims>();

  bool success = true;
  for (size_t I = 0; I < Acc.size(); ++I) {
    CoordT<ImgType, Dims> Coord =
        DelinearizeToCoord<ImgType>(I, ImageRange, Normalized);

    for (const auto &Offset : Offsets) {
      // Normalize offset if needed.
      auto AdjOffset = Normalized
                           ? Offset / RangeToCoord<ImgType, Dims>(ImageRange, 2)
                           : Offset;
      auto OffsetCoord = Coord + AdjOffset;

      float4 ReadVal = Acc.read(OffsetCoord).template convert<float>();
      float4 ExpectedVal = CalcLinearRead<Format, AddrMode, Dims>(
          RefData, OffsetCoord, ImagePitch, ImageRange, Normalized);

      // Compare results as floats to better check for precision differences.
      // I.e. if the type is already represented as float we can simply use a
      // small precision, while if they are integral the precision could have
      // affected the rounding.
      float Precision =
          std::is_integral_v<typename FormatTraits<Format>::rep_elem_type>
              ? 0.5001
              : 0.1;
      if (!ApproxEq(ReadVal, ExpectedVal, Precision)) {
        std::cout << "Unexpected read value (" << ReadVal
                  << " != " << ExpectedVal << ") at coordinate " << OffsetCoord
                  << " (" << FormatTraits<Format>::Name << ") ("
                  << AddressingModeToString<AddrMode>() << ")" << std::endl;
        success = false;
      }
    }
  }
  return success;
}

template <image_format Format, int Dims, addressing_mode AddrMode,
          coordinate_normalization_mode CoordNormMode>
int check(std::vector<typename FormatTraits<Format>::rep_elem_type> &Data) {
  range<Dims> ImageRange =
      CreateImageRange<Dims>(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);

  constexpr image_sampler Sampler{AddrMode, CoordNormMode,
                                  filtering_mode::linear};

  int Failures = 0;

  // Test image without explicit pitch.
  sampled_image<Dims> Img1{Data.data(), Format, Sampler, ImageRange};
  Failures += !checkSampledImageHostReadLinear<Format, AddrMode, CoordNormMode>(
      Img1, Data.data());

  // If Dims > 1 test image image with a pitch different than the image size.
  if constexpr (Dims > 1) {
    constexpr size_t REP_ELEM_VEC_SIZE =
        sizeof(typename FormatTraits<Format>::rep_elem_type) * 4;
    constexpr size_t IMAGE_PITCH_WIDTH_BYTES =
        IMAGE_PITCH_WIDTH * REP_ELEM_VEC_SIZE;
    constexpr size_t IMAGE_PITCH_HEIGHT_BYTES =
        IMAGE_PITCH_HEIGHT * REP_ELEM_VEC_SIZE;
    range<Dims - 1> ImagePitch = CreateImageRange<Dims - 1>(
        IMAGE_PITCH_WIDTH_BYTES, IMAGE_PITCH_HEIGHT_BYTES, 0);

    sampled_image<Dims> Img2{Data.data(), Format, Sampler, ImageRange,
                             ImagePitch};
    Failures +=
        !checkSampledImageHostReadLinear<Format, AddrMode, CoordNormMode>(
            Img2, Data.data());
  }

  return Failures;
}

template <image_format Format, int Dims>
int checkForFormatAndDims(
    std::vector<typename FormatTraits<Format>::rep_elem_type> &Data) {
  int Failures = 0;
  Failures += check<Format, Dims, addressing_mode::none,
                    coordinate_normalization_mode::unnormalized>(Data);
  Failures += check<Format, Dims, addressing_mode::clamp_to_edge,
                    coordinate_normalization_mode::unnormalized>(Data);
  Failures += check<Format, Dims, addressing_mode::clamp,
                    coordinate_normalization_mode::unnormalized>(Data);
  Failures += check<Format, Dims, addressing_mode::none,
                    coordinate_normalization_mode::normalized>(Data);
  Failures += check<Format, Dims, addressing_mode::repeat,
                    coordinate_normalization_mode::normalized>(Data);
  Failures += check<Format, Dims, addressing_mode::mirrored_repeat,
                    coordinate_normalization_mode::normalized>(Data);
  Failures += check<Format, Dims, addressing_mode::clamp_to_edge,
                    coordinate_normalization_mode::normalized>(Data);
  Failures += check<Format, Dims, addressing_mode::clamp,
                    coordinate_normalization_mode::normalized>(Data);
  return Failures;
}

template <image_format Format> int checkForFormat() {
  auto Data = GenerateData<Format>(IMAGE_PITCH_WIDTH * IMAGE_PITCH_HEIGHT *
                                   IMAGE_DEPTH);
  int Failures = 0;
  Failures += checkForFormatAndDims<Format, 1>(Data);
  Failures += checkForFormatAndDims<Format, 2>(Data);
  Failures += checkForFormatAndDims<Format, 3>(Data);
  return Failures;
}

int main() {
  int Failures = 0;
  Failures += checkForFormat<image_format::r8g8b8a8_unorm>();
  Failures += checkForFormat<image_format::r16g16b16a16_unorm>();
  Failures += checkForFormat<image_format::r8g8b8a8_sint>();
  Failures += checkForFormat<image_format::r16g16b16a16_sint>();
  Failures += checkForFormat<image_format::r32b32g32a32_sint>();
  Failures += checkForFormat<image_format::r8g8b8a8_uint>();
  Failures += checkForFormat<image_format::r16g16b16a16_uint>();
  Failures += checkForFormat<image_format::r32b32g32a32_uint>();
  Failures += checkForFormat<image_format::r16b16g16a16_sfloat>();
  Failures += checkForFormat<image_format::r32g32b32a32_sfloat>();
  Failures += checkForFormat<image_format::b8g8r8a8_unorm>();
  return Failures;
}
