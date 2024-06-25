#pragma once
#include <random>
#include <sycl/detail/core.hpp>

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

}; // namespace bindless_helpers
