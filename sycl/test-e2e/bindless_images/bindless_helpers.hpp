#pragma once
#include <random>
#include <sycl/sycl.hpp>

namespace bindless_helpers {

template <typename DType, int NChannels>
static void fill_rand(std::vector<sycl::vec<DType, NChannels>> &v,
                      int seed = std::default_random_engine::default_seed) {
  std::default_random_engine generator;
  generator.seed(seed);
  auto distribution = [&]() {
    if constexpr (std::is_same_v<DType, sycl::half>) {
      return std::uniform_real_distribution<float>(0.0, 100.0);
    } else if constexpr (std::is_floating_point_v<DType>) {
      return std::uniform_real_distribution<DType>(0.0, 100.0);
    } else {
      return std::uniform_int_distribution<DType>(0, 100);
    }
  }();
  for (int i = 0; i < v.size(); ++i) {
    sycl::vec<DType, NChannels> temp;

    for (int j = 0; j < NChannels; j++) {
      temp[j] = distribution(generator);
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

}; // namespace bindless_helpers
