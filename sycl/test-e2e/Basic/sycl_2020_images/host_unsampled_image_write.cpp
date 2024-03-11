// RUN: %{build} -o %t.out
// TODO: Consider moving to sycl/test as this is device-independent.
// RUN: %{run-unfiltered-devices} %t.out

// Tests for direct read of unsampled_image using host accessors.

#include "common.hpp"

constexpr size_t IMAGE_WIDTH = 5;
constexpr size_t IMAGE_HEIGHT = 4;
constexpr size_t IMAGE_DEPTH = 2;

constexpr size_t IMAGE_PITCH_WIDTH = 7;
constexpr size_t IMAGE_PITCH_HEIGHT = 5 * IMAGE_PITCH_WIDTH;

template <image_format Format, int Dims>
bool checkUnsampledImageHostTaskWriteDirect(unsampled_image<Dims> &Image) {
  using PixelType = typename FormatTraits<Format>::pixel_type;
  constexpr ImageType ImgType = ImageType::Unsampled;

  host_unsampled_image_accessor<PixelType, Dims, access_mode::read_write> Acc(
      Image);
  size_t AccSize = Acc.size();
  assert(Image.size() == AccSize);

  range<Dims> ImageRange = Image.get_range();

  bool success = true;
  for (size_t I = 0; I < AccSize; ++I) {
    auto Coord = DelinearizeToCoord<ImgType>(I, ImageRange);
    PixelType NewColor = PickNewColor<Format>(I, AccSize);

    // Write the new color to the coordinate, then try to read it and check that
    // it has changed accordingly.
    Acc.write(Coord, NewColor);
    PixelType ReadVal = Acc.read(Coord);
    if (!ApproxEq(ReadVal, NewColor)) {
      std::cout << "Unexpected written value (" << ReadVal << " != " << NewColor
                << ") at coordinate " << Coord << " ("
                << FormatTraits<Format>::Name << ")" << std::endl;
      success = false;
    }
  }
  return success;
}

template <image_format Format, int Dims>
int check(std::vector<typename FormatTraits<Format>::rep_elem_type> &Data) {
  range<Dims> ImageRange =
      CreateImageRange<Dims>(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);

  int Failures = 0;

  // Test image without explicit pitch.
  std::fill(Data.begin(), Data.end(), 0);
  unsampled_image<Dims> Img1{Data.data(), Format, ImageRange};
  Failures += !checkUnsampledImageHostTaskWriteDirect<Format>(Img1);

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

    std::fill(Data.begin(), Data.end(), 0);
    unsampled_image<Dims> Img2{Data.data(), Format, ImageRange, ImagePitch};
    Failures += !checkUnsampledImageHostTaskWriteDirect<Format>(Img2);
  }

  return Failures;
}

template <image_format Format> int checkForFormat() {

  auto Data = GenerateData<Format>(IMAGE_PITCH_WIDTH * IMAGE_PITCH_HEIGHT *
                                   IMAGE_DEPTH);
  int Failures = 0;
  Failures += check<Format, 1>(Data);
  Failures += check<Format, 2>(Data);
  Failures += check<Format, 3>(Data);
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
