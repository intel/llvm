#pragma once

// Shared by the D3D12 and Vulkan interop read/write tests: channel formats,
// texture contents, and a kernel that reads an imported image and writes it in
// place.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/half_type.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace interop_read_write {

namespace syclexp = sycl::ext::oneapi::experimental;

enum class Encoding { Float, UNorm, SNorm };

struct ChannelFormat {
  Encoding encoding;
  int bits; // per channel
  int channels;
  sycl::image_channel_type channelType;

  int channelBytes() const { return bits / 8; }
  uint32_t mask() const { return bits == 32 ? 0xFFFFFFFFu : (1u << bits) - 1; }
  // Raw value of 1.0 in the normalized encodings
  int normMax() const {
    return encoding == Encoding::UNorm ? (1 << bits) - 1
                                       : (1 << (bits - 1)) - 1;
  }
};

// Types: float, half, unorm8, snorm8, unorm16, snorm16
inline std::optional<ChannelFormat> getChannelFormat(const std::string &type,
                                                     int channels) {
  using CT = sycl::image_channel_type;
  if (type == "float")
    return ChannelFormat{Encoding::Float, 32, channels, CT::fp32};
  if (type == "half")
    return ChannelFormat{Encoding::Float, 16, channels, CT::fp16};
  if (type == "unorm8")
    return ChannelFormat{Encoding::UNorm, 8, channels, CT::unorm_int8};
  if (type == "snorm8")
    return ChannelFormat{Encoding::SNorm, 8, channels, CT::snorm_int8};
  if (type == "unorm16")
    return ChannelFormat{Encoding::UNorm, 16, channels, CT::unorm_int16};
  if (type == "snorm16")
    return ChannelFormat{Encoding::SNorm, 16, channels, CT::snorm_int16};
  return std::nullopt;
}

inline uint32_t encodeFloat(const ChannelFormat &f, float v) {
  if (f.bits == 32) {
    uint32_t raw;
    std::memcpy(&raw, &v, sizeof(raw));
    return raw;
  }
  const sycl::half h(v);
  uint16_t raw;
  std::memcpy(&raw, &h, sizeof(raw));
  return raw;
}

// Value an image read returns for a raw channel value
inline float decode(const ChannelFormat &f, uint32_t raw) {
  switch (f.encoding) {
  case Encoding::Float:
    if (f.bits == 32) {
      float v;
      std::memcpy(&v, &raw, sizeof(v));
      return v;
    } else {
      const uint16_t raw16 = uint16_t(raw);
      sycl::half h;
      std::memcpy(&h, &raw16, sizeof(h));
      return float(h);
    }
  case Encoding::UNorm:
    return float(double(raw) / f.normMax());
  case Encoding::SNorm: {
    int v = int(raw);
    if (v & (1 << (f.bits - 1)))
      v -= 1 << f.bits;
    return float(std::max(double(v) / f.normMax(), -1.0));
  }
  }
  return 0.f;
}

inline uint32_t mix(uint32_t v) {
  v ^= v >> 16;
  v *= 0x7feb352du;
  v ^= v >> 15;
  v *= 0x846ca68bu;
  v ^= v >> 16;
  return v;
}

// Raw channel value of the copied texture contents. Every other 32x32 tile is
// constant (compressible), the others are noise.
inline uint32_t patternValue(const ChannelFormat &f, int x, int y, int c) {
  const bool constTile = (x / 32 + y / 32) % 2 == 0;
  const uint32_t seed = constTile ? (uint32_t(y / 32) << 16 | uint32_t(x / 32))
                                  : (uint32_t(y) << 16 | uint32_t(x));
  const uint32_t h = mix(seed * 4 + c + (constTile ? 0x80000000u : 0));
  switch (f.encoding) {
  case Encoding::Float:
    // Exactly representable as half
    return encodeFloat(f, float(int(h % 2001) - 1000) / 8.f);
  case Encoding::UNorm:
    return h % uint32_t(f.normMax() + 1);
  case Encoding::SNorm:
    // Excludes the most negative value, which also reads as -1
    return uint32_t(int(h % uint32_t(2 * f.normMax() + 1)) - f.normMax()) &
           f.mask();
  }
  return 0;
}

inline constexpr float clearColor[4] = {0.25f, 0.5f, 0.75f, 1.f};

// Raw channel value of the cleared texture contents
inline uint32_t clearValue(const ChannelFormat &f, int c) {
  if (f.encoding == Encoding::Float)
    return encodeFloat(f, clearColor[c]);
  return uint32_t(std::lround(clearColor[c] * f.normMax())) & f.mask();
}

// Raw value after the kernel wrote 1 - v (unorm) or -v (float and snorm)
inline uint32_t transformedValue(const ChannelFormat &f, uint32_t raw) {
  switch (f.encoding) {
  case Encoding::Float:
    return raw ^ (1u << (f.bits - 1));
  case Encoding::UNorm:
    return uint32_t(f.normMax()) - raw;
  case Encoding::SNorm:
    return (0u - raw) & f.mask();
  }
  return 0;
}

// Writes the pattern to rows of tightly packed pixels rowPitch bytes apart
inline void packPattern(const ChannelFormat &f, int width, int height,
                        uint8_t *dst, size_t rowPitch) {
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < f.channels; ++c) {
        const uint32_t raw = patternValue(f, x, y, c);
        std::memcpy(dst + y * rowPitch +
                        (size_t(x) * f.channels + c) * f.channelBytes(),
                    &raw, f.channelBytes());
      }
}

inline std::vector<uint32_t> unpack(const ChannelFormat &f, int width,
                                    int height, const uint8_t *src,
                                    size_t rowPitch) {
  std::vector<uint32_t> raw(size_t(width) * height * f.channels, 0);
  for (int y = 0; y < height; ++y)
    for (int i = 0; i < width * f.channels; ++i)
      std::memcpy(&raw[size_t(y) * width * f.channels + i],
                  src + y * rowPitch + size_t(i) * f.channelBytes(),
                  f.channelBytes());
  return raw;
}

template <int Channels>
using PixelT =
    std::conditional_t<Channels == 1, float, sycl::vec<float, Channels>>;

template <int Channels>
sycl::event readModifyWriteImpl(sycl::queue &q,
                                syclexp::unsampled_image_handle image,
                                int width, int height, bool unorm,
                                float *fetched, sycl::event dependency) {
  return q.submit([&](sycl::handler &h) {
    h.depends_on(dependency);
    // Bindless images use (x, y) order, ranges (y, x)
    h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
      const int x = int(id[1]);
      const int y = int(id[0]);
      using T = PixelT<Channels>;
      const T v = syclexp::fetch_image<T>(image, sycl::int2(x, y));
      float *out = fetched + (size_t(y) * width + x) * Channels;
      if constexpr (Channels == 1)
        out[0] = v;
      else
        for (int c = 0; c < Channels; ++c)
          out[c] = v[c];
      syclexp::write_image(image, sycl::int2(x, y), unorm ? T(1.f) - v : -v);
    });
  });
}

// Reads every pixel into fetched (width * height * channels floats) and
// writes back 1 - v (unorm) or -v (float and snorm)
inline sycl::event readModifyWrite(sycl::queue &q, const ChannelFormat &f,
                                   syclexp::unsampled_image_handle image,
                                   int width, int height, float *fetched,
                                   sycl::event dependency = {}) {
  const bool unorm = f.encoding == Encoding::UNorm;
  if (f.channels == 1)
    return readModifyWriteImpl<1>(q, image, width, height, unorm, fetched,
                                  dependency);
  if (f.channels == 2)
    return readModifyWriteImpl<2>(q, image, width, height, unorm, fetched,
                                  dependency);
  return readModifyWriteImpl<4>(q, image, width, height, unorm, fetched,
                                dependency);
}

// Checks what the kernel read and the raw contents after it wrote. Returns the
// number of mismatching channel values.
inline size_t checkResults(const ChannelFormat &f, int width, int height,
                           bool clear, const float *fetched,
                           const std::vector<uint32_t> &written) {
  size_t errors = 0;
  for (size_t i = 0; i < written.size(); ++i) {
    const int c = int(i % f.channels);
    const int x = int(i / f.channels % width);
    const int y = int(i / f.channels / width);
    const uint32_t original =
        clear ? clearValue(f, c) : patternValue(f, x, y, c);
    const float expectedRead = decode(f, original);
    const uint32_t expectedWrite = transformedValue(f, original);
    const bool readOk =
        std::fabs(fetched[i] - expectedRead) <= std::fabs(expectedRead) * 1e-6f;
    if (!readOk || written[i] != expectedWrite) {
      if (errors < 10)
        std::cerr << "Mismatch at x:" << x << " y:" << y << " c:" << c
                  << " | SYCL read: " << fetched[i]
                  << " expected: " << expectedRead << " | written: 0x"
                  << std::hex << written[i] << " expected: 0x" << expectedWrite
                  << std::dec << std::endl;
      ++errors;
    }
  }
  return errors;
}

} // namespace interop_read_write
