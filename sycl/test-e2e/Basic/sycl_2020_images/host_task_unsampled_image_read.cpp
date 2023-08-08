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
bool checkUnsampledImageHostTaskReadDirect(
    unsampled_image<Dims> &Image,
    typename FormatTraits<Format>::rep_elem_type *RefData, queue &Q) {
  using PixelType = typename FormatTraits<Format>::pixel_type;
  constexpr ImageType ImgType = ImageType::Unsampled;

  bool Success = true;
  {
    buffer<bool, 1> SuccessBuf{&Success, 1};
    Q.submit([&](handler &CGH) {
      unsampled_image_accessor<PixelType, Dims, access_mode::read,
                               image_target::host_task>
          Acc(Image, CGH);
      accessor SuccessAcc{SuccessBuf, CGH, write_only_host_task};

      assert(Image.size() == Acc.size());

      CGH.host_task([=]() {
        range<Dims> ImageRange = Image.get_range();
        range<2> ImagePitch = getElementWisePitch<Format, Dims>(Image);

        for (size_t I = 0; I < Acc.size(); ++I) {
          auto Coord = DelinearizeToCoord<ImgType>(I, ImageRange);

          // Read the coordinate through the accessor and read the corresponding
          // value in the reference memory.
          PixelType ReadVal = Acc.read(Coord);
          PixelType ExpectedVal = SimulateRead<Format, ImgType>(
              RefData, Coord, ImagePitch, ImageRange);
          if (!AllTrue(ReadVal == ExpectedVal)) {
            std::cout << "Unexpected read value (" << ReadVal
                      << " != " << ExpectedVal << ") at coordinate " << Coord
                      << " (" << FormatTraits<Format>::Name << ")" << std::endl;
            SuccessAcc[0] = false;
          }
        }
      });
    });
  }
  return Success;
}

template <image_format Format, int Dims>
int check(std::vector<typename FormatTraits<Format>::rep_elem_type> &Data,
          queue &Q) {
  range<Dims> ImageRange =
      CreateImageRange<Dims>(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);

  int Failures = 0;

  // Test image without explicit pitch.
  unsampled_image<Dims> Img1{Data.data(), Format, ImageRange};
  Failures +=
      !checkUnsampledImageHostTaskReadDirect<Format>(Img1, Data.data(), Q);

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

    unsampled_image<Dims> Img2{Data.data(), Format, ImageRange, ImagePitch};
    Failures +=
        !checkUnsampledImageHostTaskReadDirect<Format>(Img2, Data.data(), Q);
  }

  return Failures;
}

template <image_format Format> int checkForFormat(queue &Q) {
  auto Data = GenerateData<Format>(IMAGE_PITCH_WIDTH * IMAGE_PITCH_HEIGHT *
                                   IMAGE_DEPTH);
  int Failures = 0;
  Failures += check<Format, 1>(Data, Q);
  Failures += check<Format, 2>(Data, Q);
  Failures += check<Format, 3>(Data, Q);
  return Failures;
}

int main() {
  queue Q;
  int Failures = 0;
  Failures += checkForFormat<image_format::r8g8b8a8_unorm>(Q);
  Failures += checkForFormat<image_format::r16g16b16a16_unorm>(Q);
  Failures += checkForFormat<image_format::r8g8b8a8_sint>(Q);
  Failures += checkForFormat<image_format::r16g16b16a16_sint>(Q);
  Failures += checkForFormat<image_format::r32b32g32a32_sint>(Q);
  Failures += checkForFormat<image_format::r8g8b8a8_uint>(Q);
  Failures += checkForFormat<image_format::r16g16b16a16_uint>(Q);
  Failures += checkForFormat<image_format::r32b32g32a32_uint>(Q);
  Failures += checkForFormat<image_format::r16b16g16a16_sfloat>(Q);
  Failures += checkForFormat<image_format::r32g32b32a32_sfloat>(Q);
  Failures += checkForFormat<image_format::b8g8r8a8_unorm>(Q);
  return Failures;
}
