// RUN: %{build} -o %t.out
// TODO: Consider moving to sycl/test as this is device-independent.
// RUN: %{run-unfiltered-devices} %t.out

// Tests the members of the unsampled_image class.

#include "common.hpp"

constexpr size_t IMAGE_WIDTH = 5;
constexpr size_t IMAGE_HEIGHT = 4;
constexpr size_t IMAGE_DEPTH = 2;

constexpr size_t IMAGE_PITCH_WIDTH = 7;
constexpr size_t IMAGE_PITCH_HEIGHT = 5 * IMAGE_PITCH_WIDTH;

using namespace sycl;

template <image_format Format, typename T>
int checkEqual(const T &Lhs, const T &Rhs, std::string_view ErrorStr) {
  bool Failed = Lhs != Rhs;
  if (Failed)
    std::cerr << ErrorStr << " (" << Lhs << " != " << Rhs << ")  ("
              << FormatTraits<Format>::Name << ")" << std::endl;
  return Failed;
}

template <image_format Format, bool ExplicitPitch, int Dims, typename AllocT>
int check(const unsampled_image<Dims, AllocT> &Img) {
  int Failures = 0;

  // Check get_range().
  range<Dims> ExpectedRange =
      CreateImageRange<Dims>(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);
  Failures += checkEqual<Format>(Img.get_range(), ExpectedRange,
                                 "Unexpected value returned by get_range()");

  // Check size().
  size_t ExpectedSize = ExpectedRange.size();
  Failures += checkEqual<Format>(Img.size(), ExpectedSize,
                                 "Unexpected value returned by size()");

  if constexpr (Dims > 1) {
    // Check get_pitch().
    range<Dims - 1> ExpectedPitch = [&]() {
      if constexpr (ExplicitPitch)
        return CreateImageRange<Dims - 1>(IMAGE_PITCH_WIDTH, IMAGE_PITCH_HEIGHT,
                                          0);
      else
        return CreateImageRange<Dims - 1>(IMAGE_WIDTH,
                                          IMAGE_WIDTH * IMAGE_HEIGHT, 0);
    }();
    ExpectedPitch *= BytesPerPixel<Format>;
    Failures += checkEqual<Format>(Img.get_pitch(), ExpectedPitch,
                                   "Unexpected value returned by get_pitch()");

    // Check byte_size() for cases with a pitch.
    size_t ExpectedByteSize = ExpectedPitch[Dims - 2] * ExpectedRange[Dims - 1];
    Failures += checkEqual<Format>(Img.byte_size(), ExpectedByteSize,
                                   "Unexpected value returned by byte_size()");
  } else {
    // Check byte_size() for cases without a pitch.
    size_t ExpectedByteSize = ExpectedSize * BytesPerPixel<Format>;
    Failures += checkEqual<Format>(Img.byte_size(), ExpectedByteSize,
                                   "Unexpected value returned by byte_size()");
  }

  // Check return type of get_allocator().
  if (!std::is_same_v<decltype(Img.get_allocator()), AllocT>) {
    std::cerr << "Unexpected get_allocator() return type ("
              << FormatTraits<Format>::Name << ")" << std::endl;
    ++Failures;
  }

  return Failures;
}

template <image_format Format, int Dims>
int checkFormatAndDims(std::shared_ptr<void> HostData) {
  using alternate_alloc_t = std::allocator<char>;

  int Failures = 0;

  range<Dims> ImgRange =
      CreateImageRange<Dims>(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);
  alternate_alloc_t ImgAlloc;
  Failures += check<Format, false>(unsampled_image<Dims>{Format, ImgRange});
  Failures += check<Format, false>(
      unsampled_image<Dims>{HostData.get(), Format, ImgRange});
  Failures +=
      check<Format, false>(unsampled_image<Dims>{HostData, Format, ImgRange});
  Failures += check<Format, false>(
      unsampled_image<Dims, alternate_alloc_t>{Format, ImgRange, ImgAlloc});
  Failures += check<Format, false>(unsampled_image<Dims, alternate_alloc_t>{
      HostData.get(), Format, ImgRange, ImgAlloc});
  Failures += check<Format, false>(unsampled_image<Dims, alternate_alloc_t>{
      HostData, Format, ImgRange, ImgAlloc});

  if constexpr (Dims > 1) {
    range<Dims - 1> ImgPitch =
        CreateImageRange<Dims - 1>(IMAGE_PITCH_WIDTH, IMAGE_PITCH_HEIGHT, 0) *
        BytesPerPixel<Format>;
    Failures +=
        check<Format, true>(unsampled_image<Dims>{Format, ImgRange, ImgPitch});
    Failures += check<Format, true>(
        unsampled_image<Dims>{HostData.get(), Format, ImgRange, ImgPitch});
    Failures += check<Format, true>(
        unsampled_image<Dims>{HostData, Format, ImgRange, ImgPitch});
    Failures += check<Format, true>(unsampled_image<Dims, alternate_alloc_t>{
        Format, ImgRange, ImgPitch, ImgAlloc});
    Failures += check<Format, true>(unsampled_image<Dims, alternate_alloc_t>{
        HostData.get(), Format, ImgRange, ImgPitch, ImgAlloc});
    Failures += check<Format, true>(unsampled_image<Dims, alternate_alloc_t>{
        HostData, Format, ImgRange, ImgPitch, ImgAlloc});
  }
  return Failures;
}

template <image_format Format> int checkFormat() {
  // Allocate memory to use as host-pointer. We do not rely on the data inside
  // for testing.
  using rep_elem_type = typename FormatTraits<Format>::rep_elem_type;
  std::shared_ptr<rep_elem_type> Data(
      new rep_elem_type[IMAGE_PITCH_WIDTH * IMAGE_PITCH_HEIGHT * IMAGE_DEPTH *
                        4],
      std::default_delete<rep_elem_type[]>());

  int Failures = 0;
  Failures += checkFormatAndDims<Format, 1>(Data);
  Failures += checkFormatAndDims<Format, 2>(Data);
  Failures += checkFormatAndDims<Format, 3>(Data);
  return Failures;
}

int main() {
  int Failures = 0;
  Failures += checkFormat<image_format::r8g8b8a8_unorm>();
  Failures += checkFormat<image_format::r16g16b16a16_unorm>();
  Failures += checkFormat<image_format::r8g8b8a8_sint>();
  Failures += checkFormat<image_format::r16g16b16a16_sint>();
  Failures += checkFormat<image_format::r32b32g32a32_sint>();
  Failures += checkFormat<image_format::r8g8b8a8_uint>();
  Failures += checkFormat<image_format::r16g16b16a16_uint>();
  Failures += checkFormat<image_format::r32b32g32a32_uint>();
  Failures += checkFormat<image_format::r16b16g16a16_sfloat>();
  Failures += checkFormat<image_format::r32g32b32a32_sfloat>();
  Failures += checkFormat<image_format::b8g8r8a8_unorm>();
  return Failures;
}
