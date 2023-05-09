// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %BE_RUN_PLACEHOLDER %t.out

// Tests that host-side SYCL 1.2.1 image accessors correctly read the right
// value when accessing using linear sampling mode.

#include <sycl/sycl.hpp>

using namespace sycl;

using pixel_t = sycl::int4;

constexpr size_t IMG_WIDTH = 3;
constexpr size_t IMG_HEIGHT = 3;
constexpr size_t IMG_DEPTH = 3;

pixel_t ImgData[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH] = {
    {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
     {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}},
    {{{25, 26, 27, 28}, {29, 30, 31, 32}, {33, 34, 35, 36}},
     {{37, 38, 39, 40}, {41, 42, 43, 44}, {45, 46, 47, 48}}}};

template <typename T, int Dims> bool AllTrue(const vec<T, Dims> &Vec) {
  for (size_t I = 0; I < Dims; ++I)
    if (!Vec[I])
      return false;
  return true;
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

template <typename T, int Dims>
int checkValues(const vec<T, Dims> &Actual, const vec<T, Dims> &Expected,
                std::string_view AdditionalInfo) {
  if (AllTrue(Actual == Expected))
    return 0;
  std::cout << "Value check failed! (" << Actual << " != " << Expected << ") - "
            << AdditionalInfo << std::endl;
  return 1;
}

int test3D(coordinate_normalization_mode NormMode, addressing_mode AddrMode) {
  image<3> Img(ImgData, image_channel_order::rgba,
               image_channel_type::signed_int32,
               sycl::range<3>{IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH});
  sampler Sampler(NormMode, AddrMode, filtering_mode::linear);
  auto ImgAcc = Img.get_access<pixel_t, access::mode::read>();
  float4 CoordNormFactor = NormMode == coordinate_normalization_mode::normalized
                               ? float4{IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH, 1.0f}
                               : float4{1.0f};

  int Failures = 0;

  auto CalcLinearInterp = [&](float4 C) -> pixel_t {
    int4 Prev = sycl::floor(C).convert<int>();
    int4 Next = sycl::ceil(C).convert<int>();
    float4 CA = C - Prev.convert<float>();
    auto CA000 = ImgData[Prev[2]][Prev[1]][Prev[0]].convert<float>() * CA[0] *
                 CA[1] * CA[2];
    auto CA100 = ImgData[Prev[2]][Prev[1]][Next[0]].convert<float>() *
                 (1 - CA[0]) * CA[1] * CA[2];
    auto CA010 = ImgData[Prev[2]][Next[1]][Prev[0]].convert<float>() * CA[0] *
                 (1 - CA[1]) * CA[2];
    auto CA110 = ImgData[Prev[2]][Next[1]][Next[0]].convert<float>() *
                 (1 - CA[0]) * (1 - CA[1]) * CA[2];
    auto CA001 = ImgData[Next[2]][Prev[1]][Prev[0]].convert<float>() * CA[0] *
                 CA[1] * (1 - CA[2]);
    auto CA101 = ImgData[Next[2]][Prev[1]][Next[0]].convert<float>() *
                 (1 - CA[0]) * CA[1] * (1 - CA[2]);
    auto CA011 = ImgData[Next[2]][Next[1]][Prev[0]].convert<float>() * CA[0] *
                 (1 - CA[1]) * (1 - CA[2]);
    auto CA111 = ImgData[Next[2]][Next[1]][Next[0]].convert<float>() *
                 (1 - CA[0]) * (1 - CA[1]) * (1 - CA[2]);
    return (CA000 + CA100 + CA010 + CA110 + CA001 + CA101 + CA011 + CA111)
        .convert<int>();
  };

  for (size_t X = 0; X < IMG_WIDTH - 1; ++X) {
    for (size_t Y = 0; Y < IMG_HEIGHT - 1; ++Y) {
      for (size_t Z = 0; Z < IMG_DEPTH - 1; ++Z) {
        for (float OffsetX : {0.0f, 0.25f, 0.50f, 0.75f}) {
          for (float OffsetY : {0.0f, 0.25f, 0.50f, 0.75f}) {
            for (float OffsetZ : {0.0f, 0.25f, 0.50f, 0.75f}) {
              float4 C{(float)X + OffsetX, (float)Y + OffsetY,
                       (float)Z + OffsetZ, 0};
              float4 CNorm = C / CoordNormFactor;
              Failures += checkValues(
                  ImgAcc.read(CNorm, Sampler), CalcLinearInterp(C),
                  "Index {" + std::to_string(C[0]) + "," +
                      std::to_string(C[1]) + "," + std::to_string(C[2]) + "}");
            }
          }
        }
      }
    }
  }

  return Failures;
}

int test2D(coordinate_normalization_mode NormMode, addressing_mode AddrMode) {
  image<2> Img(ImgData[0], image_channel_order::rgba,
               image_channel_type::signed_int32,
               sycl::range<2>{IMG_WIDTH, IMG_HEIGHT});
  sampler Sampler(NormMode, AddrMode, filtering_mode::linear);
  auto ImgAcc = Img.get_access<pixel_t, access::mode::read>();
  float2 CoordNormFactor = NormMode == coordinate_normalization_mode::normalized
                               ? float2{IMG_WIDTH, IMG_HEIGHT}
                               : float2{1.0f};

  int Failures = 0;

  auto CalcLinearInterp = [&](float2 C) -> pixel_t {
    int2 Prev = sycl::floor(C).convert<int>();
    int2 Next = sycl::ceil(C).convert<int>();
    float2 CA = C - Prev.convert<float>();
    auto CA00 = ImgData[0][Prev[1]][Prev[0]].convert<float>() * CA[0] * CA[1];
    auto CA10 =
        ImgData[0][Prev[1]][Next[0]].convert<float>() * (1 - CA[0]) * CA[1];
    auto CA01 =
        ImgData[0][Next[1]][Prev[0]].convert<float>() * CA[0] * (1 - CA[1]);
    auto CA11 = ImgData[0][Next[1]][Next[0]].convert<float>() * (1 - CA[0]) *
                (1 - CA[1]);
    return (CA00 + CA10 + CA01 + CA11).convert<int>();
  };

  for (size_t X = 0; X < IMG_WIDTH - 1; ++X) {
    for (size_t Y = 0; Y < IMG_HEIGHT - 1; ++Y) {
      for (float OffsetX : {0.0f, 0.25f, 0.50f, 0.75f}) {
        for (float OffsetY : {0.0f, 0.25f, 0.50f, 0.75f}) {
          float2 C{(float)X + OffsetX, (float)Y + OffsetY};
          float2 CNorm = C / CoordNormFactor;
          Failures +=
              checkValues(ImgAcc.read(CNorm, Sampler), CalcLinearInterp(C),
                          "Index {" + std::to_string(C[0]) + "," +
                              std::to_string(C[1]) + "}");
        }
      }
    }
  }

  return Failures;
}

int test1D(coordinate_normalization_mode NormMode, addressing_mode AddrMode) {
  image<1> Img(ImgData[0][0], image_channel_order::rgba,
               image_channel_type::signed_int32, sycl::range<1>{IMG_WIDTH});
  sampler Sampler(NormMode, AddrMode, filtering_mode::linear);
  auto ImgAcc = Img.get_access<pixel_t, access::mode::read>();
  float CoordNormFactor = NormMode == coordinate_normalization_mode::normalized
                              ? float{IMG_WIDTH}
                              : 1.0f;

  int Failures = 0;

  auto CalcLinearInterp = [&](float C) -> pixel_t {
    int Prev = sycl::floor(C);
    int Next = sycl::ceil(C);
    float CA = C - (float)Prev;
    return (ImgData[0][0][Prev].convert<float>() * CA +
            ImgData[0][0][Next].convert<float>() * (1.0f - CA))
        .convert<int>();
  };

  for (size_t X = 0; X < IMG_WIDTH - 1; ++X) {
    for (float OffsetX : {0.0f, 0.25f, 0.50f, 0.75f}) {
      float C{(float)X + OffsetX};
      float CNorm = C / CoordNormFactor;
      Failures += checkValues(ImgAcc.read(CNorm, Sampler), CalcLinearInterp(C),
                              "Index " + std::to_string(C));
    }
  }

  return Failures;
}

int runTests(coordinate_normalization_mode NormMode, addressing_mode AddrMode) {
  int Failures = 0;
  Failures += test1D(NormMode, AddrMode);
  Failures += test2D(NormMode, AddrMode);
  Failures += test3D(NormMode, AddrMode);
  return Failures;
}

int main() {
  int Failures = 0;
  Failures += runTests(coordinate_normalization_mode::normalized,
                       addressing_mode::mirrored_repeat);
  Failures += runTests(coordinate_normalization_mode::normalized,
                       addressing_mode::repeat);
  Failures += runTests(coordinate_normalization_mode::normalized,
                       addressing_mode::clamp_to_edge);
  Failures += runTests(coordinate_normalization_mode::normalized,
                       addressing_mode::clamp);
  Failures += runTests(coordinate_normalization_mode::normalized,
                       addressing_mode::none);
  // Samplers with unsupported configuration of mirrored_repeat/repeat
  // filtering mode with unnormalized coordinates are not supported on host.
  Failures += runTests(coordinate_normalization_mode::unnormalized,
                       addressing_mode::clamp_to_edge);
  Failures += runTests(coordinate_normalization_mode::unnormalized,
                       addressing_mode::clamp);
  Failures += runTests(coordinate_normalization_mode::unnormalized,
                       addressing_mode::none);
  return Failures;
}
