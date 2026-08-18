// REQUIRES: aspect-usm_device_allocations

// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_bindless_images_2d_usm
// REQUIRES: aspect-ext_oneapi_bindless_sampled_image_fetch_2d_usm

// XFAIL: hip
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19957

// RUN: %{build} -o %t.out

// RUN: %{run} %t.out --type float
// RUN: %{run} %t.out --type half
// RUN: %{run} %t.out --type int32
// RUN: %{run} %t.out --type uint32
// RUN: %{run} %t.out --type int16
// RUN: %{run} %t.out --type uint16
// RUN: %{run} %t.out --type uint8
// RUN: %{run} %t.out --type int8

#include <cmath>
#include <iostream>
#include <vector>

#include <sycl/detail/core.hpp>
#include <sycl/half_type.hpp>
#include <sycl/usm.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/experimental/bindless_image_info.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

template <typename T> sycl::image_channel_type getSyclChannelType();
template <> inline sycl::image_channel_type getSyclChannelType<float>() {
  return sycl::image_channel_type::fp32;
}
template <> inline sycl::image_channel_type getSyclChannelType<int32_t>() {
  return sycl::image_channel_type::signed_int32;
}
template <> inline sycl::image_channel_type getSyclChannelType<uint32_t>() {
  return sycl::image_channel_type::unsigned_int32;
}
template <> inline sycl::image_channel_type getSyclChannelType<int16_t>() {
  return sycl::image_channel_type::signed_int16;
}
template <> inline sycl::image_channel_type getSyclChannelType<uint16_t>() {
  return sycl::image_channel_type::unsigned_int16;
}
template <> inline sycl::image_channel_type getSyclChannelType<uint8_t>() {
  return sycl::image_channel_type::unsigned_int8;
}
template <> inline sycl::image_channel_type getSyclChannelType<int8_t>() {
  return sycl::image_channel_type::signed_int8;
}
template <> inline sycl::image_channel_type getSyclChannelType<sycl::half>() {
  return sycl::image_channel_type::fp16;
}

template <typename T> int runTest() {
  const size_t width = 32, height = 16;
  constexpr int channels = 4;
  using Pixel = sycl::vec<T, channels>;

  sycl::queue q;
  sycl::device dev{q.get_device()};

  const size_t numOfElements = width * height;
  const size_t widthInBytes = width * channels * sizeof(T);

  auto devicePitchAlign =
      dev.get_info<syclexp::info::device::image_row_pitch_align>();
  auto deviceMaxPitch =
      dev.get_info<syclexp::info::device::max_image_linear_row_pitch>();

  const size_t basePitch =
      devicePitchAlign *
      std::ceil(float(widthInBytes) / float(devicePitchAlign));

  std::vector<Pixel> dataIn(numOfElements), out(numOfElements);
  for (size_t j = 0; j < height; ++j) {
    for (size_t i = 0; i < width; ++i) {
      dataIn[i + width * j] = Pixel(static_cast<T>(i + width * j + 1));
    }
  }

  for (size_t pitch :
       {basePitch, basePitch * 2, basePitch * 4, basePitch * 8}) {
    if (pitch > deviceMaxPitch) {
      std::cout << "Skipping row pitch: " << pitch
                << " exceeds device max linear row pitch: " << deviceMaxPitch
                << "\n";
      continue;
    } else {
      std::cout << "Running with pitch " << pitch << "\n";
    }

    try {
      syclexp::bindless_image_sampler samp(
          sycl::addressing_mode::clamp,
          sycl::coordinate_normalization_mode::unnormalized,
          sycl::filtering_mode::nearest);

      syclexp::image_descriptor desc(
          {width, height}, channels, getSyclChannelType<T>(),
          syclexp::image_type::standard, 1, 1, 0, pitch, 0);

      void *imgMem =
          sycl::aligned_alloc_device(devicePitchAlign, pitch * height, q);
      if (imgMem == nullptr) {
        std::cout << "Failed to allocate aligned device memory!\n";
        return 1;
      }

      for (size_t j = 0; j < height; ++j) {
        q.memcpy(static_cast<char *>(imgMem) + j * pitch,
                 dataIn.data() + j * width, widthInBytes);
      }
      q.wait_and_throw();

      syclexp::sampled_image_handle imgHandle =
          syclexp::create_image(imgMem, pitch, samp, desc, q);

      {
        sycl::buffer<Pixel, 2> buf(out.data(), sycl::range<2>{height, width});
        q.submit([&](sycl::handler &h) {
          auto outAcc = buf.template get_access<sycl::access_mode::write>(h);
          h.parallel_for(sycl::range<2>{height, width}, [=](sycl::item<2> it) {
            size_t y = it.get_id(0);
            size_t x = it.get_id(1);
            Pixel px = syclexp::fetch_image<Pixel>(imgHandle, sycl::int2(x, y));
            outAcc[sycl::id<2>{y, x}] = px;
          });
        });
      }

      syclexp::destroy_image_handle(imgHandle, q);
      sycl::free(imgMem, q.get_context());

      for (size_t i = 0; i < numOfElements; ++i) {
        for (int c = 0; c < channels; ++c) {
          if (out[i][c] != dataIn[i][c]) {
            std::cout << "Failure at pitch " << pitch << ": mismatch at index "
                      << i << " channel " << c << " expected "
                      << static_cast<double>(dataIn[i][c]) << " got "
                      << static_cast<double>(out[i][c]) << "\n";
            return 1;
          }
        }
      }
    } catch (sycl::exception &e) {
      std::cerr << "SYCL Exception: " << e.what() << std::endl;
      return 1;
    }
  }
  std::cout << "Test passed!\n";
  return 0;
}

int main(int argc, char **argv) {
  std::string type = "float";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--type" && i + 1 < argc)
      type = argv[++i];
  }

  if (type == "float")
    return runTest<float>();
  if (type == "half")
    return runTest<sycl::half>();
  if (type == "int32")
    return runTest<int32_t>();
  if (type == "uint32")
    return runTest<uint32_t>();
  if (type == "int16")
    return runTest<int16_t>();
  if (type == "uint16")
    return runTest<uint16_t>();
  if (type == "uint8")
    return runTest<uint8_t>();
  if (type == "int8")
    return runTest<int8_t>();

  std::cerr << "Unknown type: " << type << "\n";
  return 1;
}
