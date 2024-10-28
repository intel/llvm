#pragma once
#include <random>
#include <sycl/detail/core.hpp>
#include <sycl/image.hpp>

template <typename DType, int NChannels>
std::ostream &operator<<(std::ostream &os,
                         const sycl::vec<DType, NChannels> &vec) {
  std::string str{""};
  for (int i = 0; i < NChannels; ++i) {
    str += std::to_string(vec[i]) + ",";
  }
  str.pop_back();
  os << str;
  return os;
}

namespace bindless_helpers {

template <int NDims>
static void printTestName(std::string name, sycl::range<NDims> globalSize,
                          sycl::range<NDims> localSize) {
#if defined(VERBOSE_LV1) || defined(VERBOSE_LV2) || defined(VERBOSE_LV3)
  std::cout << name << "\n";
  std::cout << "Global Size: ";

  for (int i = 0; i < NDims; i++) {
    std::cout << globalSize[i] << " ";
  }

  std::cout << " Local Size: ";

  for (int i = 0; i < NDims; i++) {
    std::cout << localSize[i] << " ";
  }

  std::cout << "\n";
#endif
}

const char *channelTypeToString(sycl::image_channel_type type) {
  switch (type) {
  case sycl::image_channel_type::snorm_int8:
    return "sycl::image_channel_type::snorm_int8";
  case sycl::image_channel_type::snorm_int16:
    return "sycl::image_channel_type::snorm_int16";
  case sycl::image_channel_type::unorm_int8:
    return "sycl::image_channel_type::unorm_int8";
  case sycl::image_channel_type::unorm_int16:
    return "sycl::image_channel_type::unorm_int16";
  case sycl::image_channel_type::unorm_short_565:
    return "sycl::image_channel_type::unorm_short_565";
  case sycl::image_channel_type::unorm_short_555:
    return "sycl::image_channel_type::unorm_short_555";
  case sycl::image_channel_type::unorm_int_101010:
    return "sycl::image_channel_type::unorm_int_101010";
  case sycl::image_channel_type::signed_int8:
    return "sycl::image_channel_type::signed_int8";
  case sycl::image_channel_type::signed_int16:
    return "sycl::image_channel_type::signed_int16";
  case sycl::image_channel_type::signed_int32:
    return "sycl::image_channel_type::signed_int32";
  case sycl::image_channel_type::unsigned_int8:
    return "sycl::image_channel_type::unsigned_int8";
  case sycl::image_channel_type::unsigned_int16:
    return "sycl::image_channel_type::unsigned_int16";
  case sycl::image_channel_type::unsigned_int32:
    return "sycl::image_channel_type::unsigned_int32";
  case sycl::image_channel_type::fp16:
    return "sycl::image_channel_type::fp16";
  case sycl::image_channel_type::fp32:
    return "sycl::image_channel_type::fp32";
  default:
    std::cerr << "Unsupported image_channel_type in channelTypeToString\n";
    exit(-1);
  }
}

template <typename DType, int NChannel>
constexpr sycl::vec<DType, NChannel> init_vector(DType val) {
  if constexpr (NChannel == 1) {
    return sycl::vec<DType, NChannel>{val};
  } else if constexpr (NChannel == 2) {
    return sycl::vec<DType, NChannel>{val, val};
  } else if constexpr (NChannel == 4) {
    return sycl::vec<DType, NChannel>{val, val, val, val};
  } else {
    std::cerr << "Unsupported number of channels " << NChannel << "\n";
    exit(-1);
  }
}

template <typename DType, int NChannels>
bool equal_vec(sycl::vec<DType, NChannels> v1, sycl::vec<DType, NChannels> v2) {
  for (int i = 0; i < NChannels; ++i) {
    if (v1[i] != v2[i]) {
      return false;
    }
  }
  return true;
}

template <typename DType, int NChannels>
static void fill_rand(std::vector<sycl::vec<DType, NChannels>> &v,
                      int seed = std::default_random_engine::default_seed) {
  assert(!v.empty());
  std::default_random_engine generator;
  generator.seed(seed);
  auto distribution = [&]() {
    if constexpr (std::is_same_v<DType, sycl::half>) {
      return std::uniform_real_distribution<float>(0.0, 100.0);
    } else if constexpr (std::is_floating_point_v<DType>) {
      return std::uniform_real_distribution<DType>(0.0, 100.0);
    } else if constexpr (sizeof(DType) == 1) {
      return std::uniform_int_distribution<unsigned short>(0, 100);
    } else {
      return std::uniform_int_distribution<DType>(0, 100);
    }
  }();
  for (int i = 0; i < v.size(); ++i) {
    sycl::vec<DType, NChannels> temp;

    for (int j = 0; j < NChannels; j++) {
      temp[j] = static_cast<DType>(distribution(generator));
    }

    v[i] = temp;
  }
}

template <typename DType, int NChannels>
static void add_host(const std::vector<sycl::vec<DType, NChannels>> &in_0,
                     const std::vector<sycl::vec<DType, NChannels>> &in_1,
                     std::vector<sycl::vec<DType, NChannels>> &out) {
  for (int i = 0; i < out.size(); ++i) {
    for (int j = 0; j < NChannels; ++j) {
      out[i][j] = in_0[i][j] + in_1[i][j];
    }
  }
}

template <typename DType, int NChannels,
          typename = std::enable_if_t<NChannels == 1>>
static DType add_kernel(const DType in_0, const DType in_1) {
  return in_0 + in_1;
}

template <typename DType, int NChannels,
          typename = std::enable_if_t<(NChannels > 1)>>
static sycl::vec<DType, NChannels>
add_kernel(const sycl::vec<DType, NChannels> &in_0,
           const sycl::vec<DType, NChannels> &in_1) {
  sycl::vec<DType, NChannels> out;
  for (int i = 0; i < NChannels; ++i) {
    out[i] = in_0[i] + in_1[i];
  }
  return out;
}

template <int NDims>
static constexpr sycl::range<NDims> reverse_dims(sycl::range<NDims> input) {
  if constexpr (NDims == 3) {
    return sycl::range<NDims>(input[2], input[1], input[0]);
  } else if constexpr (NDims == 2) {
    return sycl::range<NDims>(input[1], input[0]);
  } else { // NDims == 1
    return input;
  }
}

template <int NDims> struct ImageArrayDims {
  template <int Dims = NDims, typename = std::enable_if_t<Dims == 3>>
  ImageArrayDims(sycl::range<3> dims) : array_count(dims[2]) {
    array_dims[0] = dims[0];
    array_dims[1] = dims[1];
  }

  template <int Dims = NDims, typename = std::enable_if_t<Dims == 2>>
  ImageArrayDims(sycl::range<2> dims) : array_count(dims[1]) {
    array_dims[0] = dims[0];
  }

  sycl::range<NDims - 1> array_dims;
  unsigned int array_count;
};

template <int NDims> static sycl::range<NDims> getGlobalSize(size_t index) {

  const std::vector<sycl::range<1>> globalSizes1D = {{32}, {16}, {20},
                                                     {9},  {14}, {2}};
  const std::vector<sycl::range<2>> globalSizes2D = {{32, 16}, {8, 32}, {20, 5},
                                                     {3, 9},   {14, 7}, {2, 2}};
  const std::vector<sycl::range<3>> globalSizes3D = {
      {16, 8, 4}, {2, 6, 12}, {10, 15, 5}, {9, 6, 3}, {15, 7, 3}, {2, 2, 2}};

  const size_t globalIndex = index % 6;

  if constexpr (NDims == 1) {
    return {globalSizes1D[globalIndex]};
  }

  if constexpr (NDims == 2) {
    return {globalSizes2D[globalIndex]};
  }

  if constexpr (NDims == 3) {
    return {globalSizes3D[globalIndex]};
  }
}

template <int NDims> static sycl::range<NDims> getLocalSize(size_t index) {

  const std::vector<sycl::range<1>> localSizes1D = {{2}, {16}, {5},
                                                    {3}, {7},  {1}};
  const std::vector<sycl::range<2>> localSizes2D = {{16, 4}, {2, 32}, {5, 5},
                                                    {3, 3},  {7, 7},  {1, 1}};
  const std::vector<sycl::range<3>> localSizes3D = {
      {8, 4, 2}, {1, 3, 12}, {5, 5, 5}, {3, 3, 3}, {5, 7, 3}, {1, 1, 1}};

  const size_t localIndex = index % 6;

  if constexpr (NDims == 1) {
    return localSizes1D[localIndex];
  }

  if constexpr (NDims == 2) {
    return localSizes2D[localIndex];
  }

  if constexpr (NDims == 3) {
    return localSizes3D[localIndex];
  }
}

}; // namespace bindless_helpers
