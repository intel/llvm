// REQUIRES: linux
// REQUIRES: cuda

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %t.out

#include <cassert>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

// Print test names and pass status
// #define VERBOSE_LV1

// Same as above plus sampler, offset, margin of error, largest error found and
// results of one mismatch
// #define VERBOSE_LV2

// Same as above but all mismatches are printed
// #define VERBOSE_LV3

// Helpers and utilities
struct util {
  template <typename DType, int NChannels>
  static void fill_rand(std::vector<sycl::vec<DType, NChannels>> &v, int seed) {
    std::default_random_engine generator;
    generator.seed(seed);
    auto distribution = [&]() {
      auto distr_t_zero = []() {
        if constexpr (std::is_same_v<DType, sycl::half>) {
          return float{};
        } else if constexpr (sizeof(DType) == 1) {
          return int{};
        } else {
          return DType{};
        }
      }();
      using distr_t = decltype(distr_t_zero);
      if constexpr (std::is_floating_point_v<distr_t>) {
        return std::uniform_real_distribution(distr_t_zero,
                                              static_cast<distr_t>(100));
      } else {
        return std::uniform_int_distribution<distr_t>(distr_t_zero, 100);
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

  // Returns the two pixels to access plus the weight each of them have
  static double get_common_linear_fract_and_coords_fp64(double coord, int *x0,
                                                        int *x1) {
    double pixelCoord;

    // sycl::fract stores results into a multi_ptr instead of a raw pointer.
    sycl::private_ptr<double> pPixelCoord = &pixelCoord;

    // Subtract to align so that pixel center is 0.5 away from origin.
    coord = coord - 0.5;

    double weight = sycl::fract(coord, pPixelCoord);
    *x0 = static_cast<int>(*pPixelCoord);
    *x1 = *x0 + 1;
    return weight;
  }

  // Linear sampling is the process of giving a weighted linear blend
  // between the nearest adjacent pixels.
  // When performing linear sampling, we subtract 0.5 from the original
  // coordinate to get the center-adjusted coordinate (as pixels are "centered"
  // on the half-integers). For example, with original coord 3.2, we get a
  // center-adjusted coord of 2.7. With 2.7, we have 70% of the pixel value will
  // come from the pixel at coord 3 and 30% from the pixel value at coord 2

  // The function arguments here are the two pixels to use and the weight to
  // give each of them.
  template <int NChannels, typename DType>
  static sycl::vec<DType, NChannels>
  linearOp1D(sycl::vec<DType, NChannels> pix1, sycl::vec<DType, NChannels> pix2,
             double weight) {

    sycl::vec<double, NChannels> weightArr(weight);
    sycl::vec<double, NChannels> one(1.0f);

    sycl::vec<double, NChannels> Ti0 = pix1.template convert<double>();
    sycl::vec<double, NChannels> Ti1 = pix2.template convert<double>();

    sycl::vec<double, NChannels> result;

    result = ((one - weightArr) * Ti0 + weightArr * Ti1);

    // Round to nearest whole number.
    // There is no option to do this via sycl::rounding_mode.
    if constexpr (std::is_same_v<DType, short> ||
                  std::is_same_v<DType, unsigned short> ||
                  std::is_same_v<DType, signed char> ||
                  std::is_same_v<DType, unsigned char>) {
      for (int i = 0; i < NChannels; i++) {
        result[i] = std::round(result[i]);
      }
    }

    return result.template convert<DType>();
  }

  // Out of range coords return a border color
  // The border color happens to be all zeros
  template <typename VecType>
  static VecType clampNearest(double coordX, int width,
                              std::vector<VecType> &input_image) {
    // Due to pixel centers being 0.5 away from origin and because
    // 0.5 is *not* subtracted here, rounding down gives the same results as
    // rounding to nearest number if 0.5 is subtracted to account
    // for pixel center
    int coordXInt = static_cast<int>(std::floor(coordX));

    // Clamp sampling according to the SYCL spec returns a border color.
    // The border color is all zeros.
    // There does not appear to be any way for the user to set the border color
    if (coordXInt > width - 1 || coordXInt < 0) {
      return VecType{0};
    }
    return input_image[coordXInt];
  }

  // Out of range coords are clamped to the extent.
  template <typename VecType>
  static VecType clampToEdgeNearest(double coordX, int width,
                                    std::vector<VecType> &input_image) {
    // Due to pixel centers being 0.5 away from origin and because
    // 0.5 is *not* subtracted here, rounding down gives the same results as
    // rounding to nearest number if 0.5 is subtracted to account
    // for pixel center
    int coordXInt = static_cast<int>(std::floor(coordX));
    // Clamp to extent
    coordXInt = std::clamp(coordXInt, 0, width - 1);
    return input_image[coordXInt];
  }

  // Out of range coords are wrapped to the valid range.
  template <typename VecType>
  static VecType repeatNearest(double coordX, int width,
                               std::vector<VecType> &input_image) {

    // Convert unnormalized input coord to normalized format
    double normCoordX = coordX / width;

    // Keep only the fractional component of the number and unnormalize.
    double fractComp = (normCoordX - std::floor(normCoordX));

    // Unnormalize fractComp
    double unnorm = fractComp * width;

    // Due to pixel centers being 0.5 away from origin and because
    // 0.5 is *not* subtracted here, rounding down gives the same results as
    // rounding to nearest number if 0.5 is subtracted to account
    // for pixel center
    int coordXInt = static_cast<int>(std::floor(unnorm));

    // Handle negative coords
    if (coordXInt < 0) {
      coordXInt = width + coordXInt;
    }

    return input_image[coordXInt];
  }

  // Out of range coordinates are flipped at every integer junction
  template <typename VecType>
  static VecType mirroredRepeatNearest(double coordX, int width,
                                       std::vector<VecType> &input_image) {

    // Convert unnormalized input coord to normalized format
    double normCoordX = coordX / width;

    // Round to nearest multiple of two.
    // e.g.
    // normCoordX == 0.3  -> result = 0
    // normCoordX == 1.3  -> result = 2
    // normCoordX == 2.4  -> result = 2
    // normCoordX == 3.42 -> result = 4
    double nearestMulOfTwo = 2.0f * std::rint(0.5f * normCoordX);
    // Subtract nearestMulOfTwo from normCoordX.
    // Gives the normalized form of the coord to use.
    // With normCoordX=1.3, norm is set to 0.7
    // With normCoordX=2.4, norm is set to 0.4
    double norm = std::abs(normCoordX - nearestMulOfTwo);
    // Unnormalize norm
    double unnorm = norm * width;
    // Round down and cast to int
    int coordXInt = static_cast<int>(std::floor(unnorm));
    // Constrain to valid range
    coordXInt = std::min(coordXInt, width - 1);

    // This prevents when at an integer junction, having three
    // accesses to pixel at normalized location 0 and 1 instead of two which is
    // correct.
    int oddShift = 0;
    // If not at image boundry and precisely on a pixel
    if (std::fmod(normCoordX, 1) != 0.0 &&
        std::fmod(normCoordX * width, 1) == 0.0) {
      // Set oddShift to be one when the integral part of the normalized
      // coords is odd.
      // Otherwise set to zero.
      oddShift =
          std::abs(static_cast<int>(std::fmod(std::floor(normCoordX), 2)));
    }
    coordXInt -= oddShift;

    return input_image[coordXInt];
  }

  // Out of range coords return a border color
  // The border color is all zeros
  template <typename DType, int NChannels>
  static sycl::vec<DType, NChannels>
  clampLinear(double coordX, int width,
              std::vector<sycl::vec<DType, NChannels>> &input_image) {
    using VecType = sycl::vec<DType, NChannels>;
    // Get coords for linear sampling
    int i0, i1;
    double weight = get_common_linear_fract_and_coords_fp64(coordX, &i0, &i1);

    VecType pix1;
    VecType pix2;

    // Clamp sampling according to the SYCL spec returns a border color.
    // The border color is all zeros.
    // There does not appear to be any way for the user to set the border color.
    if (i0 < 0 || i0 > width - 1) {
      pix1 = VecType(0);
    } else {
      pix1 = input_image[i0];
    }

    if (i1 < 0 || i1 > width - 1) {
      pix2 = VecType(0);
    } else {
      pix2 = input_image[i1];
    }

    // Perform linear sampling
    return linearOp1D<NChannels, DType>(pix1, pix2, weight);
  }

  // Out of range coords are clamped to the extent.
  template <typename DType, int NChannels>
  static sycl::vec<DType, NChannels>
  clampToEdgeLinear(double coordX, int width,
                    std::vector<sycl::vec<DType, NChannels>> &input_image) {
    using VecType = sycl::vec<DType, NChannels>;
    // Get coords for linear sampling
    int i0, i1;
    double weight = get_common_linear_fract_and_coords_fp64(coordX, &i0, &i1);

    // Clamp to extent
    i0 = std::clamp(i0, 0, width - 1);
    i1 = std::clamp(i1, 0, width - 1);

    VecType pix1 = input_image[i0];
    VecType pix2 = input_image[i1];

    // Perform linear sampling
    return linearOp1D<NChannels, DType>(pix1, pix2, weight);
  }

  // Out of range coords are wrapped to the valid range
  template <typename DType, int NChannels>
  static sycl::vec<DType, NChannels>
  repeatLinear(double coordX, int width,
               std::vector<sycl::vec<DType, NChannels>> &input_image) {
    using VecType = sycl::vec<DType, NChannels>;

    // Convert unnormalized input coord to normalized format
    double normCoordX = coordX / width;

    double unnorm = (normCoordX - static_cast<int>(normCoordX)) * width;
    // Get coords for linear sampling
    int i0, i1;
    double weight = get_common_linear_fract_and_coords_fp64(unnorm, &i0, &i1);

    // Wrap linear sampling coords to valid range
    if (i0 < 0) {
      i0 = width + i0;
    }
    if (i1 < 0) {
      i1 = width + i1;
    }

    if (i1 > width - 1) {
      i1 = i1 - width;
    }
    if (i0 > width - 1) {
      i0 = i0 - width;
    }

    VecType pix1 = input_image[i0];
    VecType pix2 = input_image[i1];

    // Perform linear sampling
    return linearOp1D<NChannels, DType>(pix1, pix2, weight);
  }

  // Out of range coordinates are flipped at every integer junction
  template <typename DType, int NChannels>
  static sycl::vec<DType, NChannels>
  mirroredRepeatLinear(double coordX, int width,
                       std::vector<sycl::vec<DType, NChannels>> &input_image) {
    using VecType = sycl::vec<DType, NChannels>;

    // Convert unnormalized input coord to normalized format
    double normCoordX = coordX / width;

    // Round to nearest multiple of two.
    // e.g.
    // normCoordX == 0.3  -> result = 0
    // normCoordX == 1.3  -> result = 2
    // normCoordX == 2.4  -> result = 2
    // normCoordX == 3.42 -> result = 4
    double nearestMulOfTwo = 2.0f * std::rint(0.5f * normCoordX);
    // Subtract nearestMulOfTwo from normCoordX.
    // Gives the normalized form of the coord to use.
    // With normCoordX=1.3, norm is set to 0.7
    // With normCoordX=2.4, norm is set to 0.4
    double norm = std::abs(normCoordX - nearestMulOfTwo);
    // Unnormalize norm
    double unnorm = norm * width;

    // Get coords for linear sampling
    int i0, i1;
    double weight = get_common_linear_fract_and_coords_fp64(unnorm, &i0, &i1);

    // get_common_linear sometimes returns numbers out of bounds.
    // Handle this by wrapping to boundary.
    i0 = std::max(i0, 0);
    i1 = std::min(i1, width - 1);

    VecType pix1 = input_image[i0];
    VecType pix2 = input_image[i1];

    // Perform linear sampling
    return linearOp1D<NChannels, DType>(pix1, pix2, weight);
  }

  template <int NDims, typename DType, int NChannels,
            typename = std::enable_if_t<NDims == 1>>
  static sycl::vec<DType, NChannels>
  read(sycl::range<1> globalSize, double coordX, double offset,
       sycl::ext::oneapi::experimental::bindless_image_sampler &samp,
       std::vector<sycl::vec<DType, NChannels>> &input_image) {
    using VecType = sycl::vec<DType, NChannels>;
    coordX = coordX + offset;
    int width = globalSize[0];

    // Ensure that coordX always contains unnormalized coords
    sycl::coordinate_normalization_mode SampNormMode = samp.coordinate;
    if (SampNormMode == sycl::coordinate_normalization_mode::normalized) {
      // Unnormalize
      coordX = coordX * width;
    }

    sycl::filtering_mode SampFiltMode = samp.filtering;
    if (SampFiltMode == sycl::filtering_mode::nearest) {

      sycl::addressing_mode SampAddrMode = samp.addressing;
      if (SampAddrMode == sycl::addressing_mode::clamp) {
        return clampNearest<VecType>(coordX, width, input_image);
      }

      if (SampAddrMode == sycl::addressing_mode::clamp_to_edge) {
        return clampToEdgeNearest<VecType>(coordX, width, input_image);
      }

      if (SampAddrMode == sycl::addressing_mode::repeat) {
        if (SampNormMode == sycl::coordinate_normalization_mode::unnormalized) {
          assert(false &&
                 "Repeat addressing mode must be used with normalized coords");
        }
        return repeatNearest(coordX, width, input_image);
      }

      if (SampAddrMode == sycl::addressing_mode::mirrored_repeat) {
        if (SampNormMode == sycl::coordinate_normalization_mode::unnormalized) {
          assert(false && "Mirrored repeat addressing mode must be used with "
                          "normalized coords");
        }
        return mirroredRepeatNearest(coordX, width, input_image);
      }

      if (SampAddrMode == sycl::addressing_mode::none) {
        int intCoordX = static_cast<int>(std::floor(coordX));
        if (intCoordX < 0 || intCoordX >= width) {
          assert(false && "Accessed out of bounds with addressing mode none! "
                          "Undefined Behaviour!");
        }
        return input_image[intCoordX];
      }

    } else { // linear
      sycl::addressing_mode SampAddrMode = samp.addressing;
      if (SampAddrMode == sycl::addressing_mode::clamp) {
        return clampLinear<DType, NChannels>(coordX, width, input_image);
      }
      if (SampAddrMode == sycl::addressing_mode::clamp_to_edge) {
        return clampToEdgeLinear<DType, NChannels>(coordX, width, input_image);
      }
      if (SampAddrMode == sycl::addressing_mode::repeat) {
        if (SampNormMode == sycl::coordinate_normalization_mode::unnormalized) {
          assert(false &&
                 "Repeat addressing mode must be used with normalized coords");
        }
        return repeatLinear<DType, NChannels>(coordX, width, input_image);
      }
      if (SampAddrMode == sycl::addressing_mode::mirrored_repeat) {
        if (SampNormMode == sycl::coordinate_normalization_mode::unnormalized) {
          assert(false && "Mirrored repeat addressing mode must be used with "
                          "normalized coords");
        }
        return mirroredRepeatLinear<DType, NChannels>(coordX, width,
                                                      input_image);
      }
      if (SampAddrMode == sycl::addressing_mode::none) {
        if (coordX < 0 || coordX >= width) {
          assert(false && "Accessed out of bounds with addressing mode none! "
                          "Undefined Behaviour!");
        }
        assert(false && "filtering mode linear with addressing mode none "
                        "currently not supported");
      }
    }
    assert(false && "Invalid sampler encountered!");
  }

  // parallel_for ND bound normalized
  template <int NDims, typename DType, int NChannels>
  static void run_ndim_test_host(
      sycl::range<NDims> globalSize, double offset,
      sycl::ext::oneapi::experimental::bindless_image_sampler &samp,
      std::vector<sycl::vec<DType, NChannels>> &input_image,
      std::vector<sycl::vec<DType, NChannels>> &output) {
    using VecType = sycl::vec<DType, NChannels>;
    bool isNorm =
        (samp.coordinate == sycl::coordinate_normalization_mode::normalized);

    if constexpr (NDims == 1) {
      for (int i = 0; i < globalSize[0]; i++) {
        double coordX;
        if (isNorm) {
          coordX = (double)i / (double)globalSize[0];
        } else {
          coordX = i;
        }
        VecType result = read<NDims, DType, NChannels>(
            globalSize, coordX, offset, samp, input_image);
        output[i] = result;
      }
    } else if constexpr (NDims == 2) {
      assert(false && "2d normalized not yet implemented");
    } else if constexpr (NDims == 3) {
      assert(false && "3d normalized not yet implemented");
    } else {
      assert(false && "Invalid dimension number set");
    }
  }

  // parallel_for ND bindless normalized
  template <int NDims, typename DType, int NChannels, typename KernelName>
  static void run_ndim_test_device(
      sycl::queue &q, sycl::range<NDims> globalSize,
      sycl::range<NDims> localSize, double offset,
      sycl::ext::oneapi::experimental::bindless_image_sampler &samp,
      sycl::ext::oneapi::experimental::sampled_image_handle input_image,
      sycl::buffer<sycl::vec<DType, NChannels>, NDims> &output,
      sycl::range<NDims> bufSize) {
    using VecType = sycl::vec<DType, NChannels>;
    bool isNorm =
        (samp.coordinate == sycl::coordinate_normalization_mode::normalized);
    if constexpr (NDims == 1) {
      try {
        q.submit([&](sycl::handler &cgh) {
          auto outAcc = output.template get_access<sycl::access_mode::write>(
              cgh, bufSize);
          cgh.parallel_for<KernelName>(
              sycl::nd_range<NDims>{globalSize, localSize},
              [=](sycl::nd_item<NDims> it) {
                size_t dim0 = it.get_global_id(0);
                double coordX = 0.0;
                if (isNorm) {
                  coordX = (double)dim0 / (double)globalSize[0];
                } else {
                  coordX = dim0;
                }

                VecType px1 =
                    sycl::ext::oneapi::experimental::read_image<VecType>(
                        input_image, float(coordX + offset));

                outAcc[(int)dim0] = px1;
              });
        });
      } catch (sycl::exception e) {
        std::cerr << "\tKernel submission failed! " << e.what() << std::endl;
        exit(-1);
      } catch (...) {
        std::cerr << "\tKernel submission failed!" << std::endl;
        exit(-1);
      }
    } else if constexpr (NDims == 2) {
      assert(false && "2d normalized not yet implemented");
    } else if constexpr (NDims == 3) {
      assert(false && "3d normalized not yet implemented");
    } else {
      assert(false && "Invalid dimension number set");
    }
  }
};

void printTestInfo(
    sycl::ext::oneapi::experimental::bindless_image_sampler &samp,
    double offset) {

  sycl::addressing_mode SampAddrMode = samp.addressing;
  sycl::coordinate_normalization_mode SampNormMode = samp.coordinate;
  sycl::filtering_mode SampFiltMode = samp.filtering;

  std::cout << "---------------------------------------NEW "
               "SAMPLER---------------------------------------\n";

  std::cout << "addressing mode: ";
  switch (SampAddrMode) {
  case sycl::addressing_mode::mirrored_repeat:
    std::cout << "mirrored_repeat\n";
    break;
  case sycl::addressing_mode::repeat:
    std::cout << "repeat\n";
    break;
  case sycl::addressing_mode::clamp_to_edge:
    std::cout << "clamp_to_edge\n";
    break;
  case sycl::addressing_mode::clamp:
    std::cout << "clamp\n";
    break;
  case sycl::addressing_mode::none:
    std::cout << "none\n";
    break;
  }

  std::cout << "coordinate normalization mode: ";
  switch (SampNormMode) {
  case sycl::coordinate_normalization_mode::normalized:
    std::cout << "normalized\n";
    break;
  case sycl::coordinate_normalization_mode::unnormalized:
    std::cout << "unnormalized\n";
    break;
  }

  std::cout << "filtering mode: ";
  switch (SampFiltMode) {
  case sycl::filtering_mode::nearest:
    std::cout << "nearest\n";
    break;
  case sycl::filtering_mode::linear:
    std::cout << "linear\n";
    break;
  }
  std::cout << "offset: " << offset << "\n";
}

bool isNumberWithinPercentOfNumber(float firstN, float percent, float secondN,
                                   float &diff, float &percDiff) {
  // Get absolute difference of the two numbers
  diff = std::abs(firstN - secondN);
  // Get the percentage difference of the two numbers
  percDiff =
      100.0f * (std::abs(firstN - secondN) / (((firstN + secondN) / 2.0f)));

  // Check if perc difference is not greater then maximum allowed
  return percDiff <= percent;
}

template <int NDims, typename DType, int NChannels,
          sycl::image_channel_type CType, sycl::image_channel_order COrder,
          typename KernelName1, typename KernelName2>
bool run_test(sycl::range<NDims> dims, sycl::range<NDims> localSize,
              double offset,
              sycl::ext::oneapi::experimental::bindless_image_sampler &samp,
              unsigned int seed = 0) {
  using VecType = sycl::vec<DType, NChannels>;

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // skip half tests if not supported
  if constexpr (std::is_same_v<DType, sycl::half>) {
    if (!dev.has(sycl::aspect::fp16)) {
#if defined(VERBOSE_LV1) || defined(VERBOSE_LV2) || defined(VERBOSE_LV3)
      std::cout << "Test skipped due to lack of device support for fp16\n";
#endif
      return false;
    }
  }

  size_t num_elems = dims[0];
  if (NDims > 1)
    num_elems *= dims[1];
  if (NDims > 2)
    num_elems *= dims[2];

  std::vector<VecType> input_0(num_elems);
  std::vector<VecType> expected(num_elems);
  std::vector<VecType> actual(num_elems);

  std::srand(seed);
  util::fill_rand(input_0, seed);

  {
    sycl::range<NDims> globalSize = dims;
    util::run_ndim_test_host<NDims, DType, NChannels>(globalSize, offset, samp,
                                                      input_0, expected);
  }

  try {

    sycl::ext::oneapi::experimental::image_descriptor desc(dims, COrder, CType);

    sycl::ext::oneapi::experimental::image_mem img_mem_0(desc, q);

    auto img_input =
        sycl::ext::oneapi::experimental::create_image(img_mem_0, samp, desc, q);

    q.ext_oneapi_copy(input_0.data(), img_mem_0.get_handle(), desc);
    q.wait_and_throw();

    {
      sycl::range<NDims> bufSize = dims;
      sycl::range<NDims> globalSize = dims;
      sycl::buffer<VecType, NDims> outBuf((VecType *)actual.data(), bufSize);
      q.wait_and_throw();
      util::run_ndim_test_device<NDims, DType, NChannels, KernelName1>(
          q, globalSize, localSize, offset, samp, img_input, outBuf, bufSize);
      q.wait_and_throw();
    }

    // Cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(img_input, q);

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return true;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return true;
  }

  // Collect and validate output

  // The following sets the percentage margin of error.

  // The margin of error might be different for different backends.
  // For CUDA, low-precision interpolation is used for linear sampling
  // according to the CUDA programming guide.
  // Specifically, CUDA devices uses 9 bits for the linear sampling weight with
  // 8 for the fractional value. (One extra so 1.0 is exactly represented)
  // 8 bits for the fractional value means there are 256 possible values
  // to represent between 1 and 0. As a percentage error, (1/256) * 100
  // gives 0.390625. Meaning that the percentage error for every
  // linear interpolation is up to 0.390625% away from the correct value.
  // There is no error when linear sampling does not occur.

  float deviation = 0.390625f;

  // For tests using nearest filtering mode, no margin of error is expected.
  if (samp.filtering == sycl::filtering_mode::nearest) {
    deviation = 0.0f;
  }

  bool validated = true;
  float largestError = 0.0f;
  float largestPercentError = 0.0f;
  for (int i = 0; i < num_elems; i++) {
    for (int j = 0; j < NChannels; ++j) {
      bool mismatch = false;
      if (actual[i][j] != expected[i][j]) {
        // Nvidia GPUs have a 0.4%~ margin of error due to only using 8 bits to
        // represent values between 1-0 for weights during linear interpolation.
        float diff, percDiff;
        if (!isNumberWithinPercentOfNumber(actual[i][j], deviation,
                                           expected[i][j], diff, percDiff)) {
          mismatch = true;
          validated = false;
        }
        if (diff > largestError) {
          largestError = diff;
        }
        if (percDiff > largestPercentError) {
          largestPercentError = percDiff;
        }
      }
      if (mismatch) {
#if defined(VERBOSE_LV2) || defined(VERBOSE_LV3)
        std::cout << "\tResult mismatch at [" << i << "][" << j
                  << "] Expected: " << +DType(expected[i][j])
                  << ", Actual: " << +DType(actual[i][j]) << std::endl;
#endif

#ifndef VERBOSE_LV3
        break;
#endif
      }
    }
#ifndef VERBOSE_LV3
    if (!validated) {
      break;
    }
#endif
  }

#if defined(VERBOSE_LV2) || defined(VERBOSE_LV3)
  std::cout << "largestError: " << largestError << "\n";
  std::cout << "largestPercentError: " << largestPercentError << "%"
            << "\n";
  std::cout << "Margin of Error: " << deviation << "%"
            << "\n";
#endif

#if defined(VERBOSE_LV1) || defined(VERBOSE_LV2) || defined(VERBOSE_LV3)
  if (validated) {
    std::cout << "\tTest passed!\n";
  } else {
    std::cout << "\tTest failed!\n";
  }
#endif

  return !validated;
}

void printTestName(std::string name) {
#if defined(VERBOSE_LV1) || defined(VERBOSE_LV2) || defined(VERBOSE_LV3)
  std::cout << name;
#endif
}

template <int NDims, typename = std::enable_if_t<NDims == 1>>
bool run_tests(sycl::range<NDims> dims, sycl::range<NDims> localSize,
               double offset, int seed,
               sycl::coordinate_normalization_mode normMode) {

  // addressing_mode::none currently removed due to
  // inconsistent behavour when switching between
  // normalized and unnormalized coords.
  sycl::addressing_mode addrModes[4] = {
      sycl::addressing_mode::repeat, sycl::addressing_mode::mirrored_repeat,
      sycl::addressing_mode::clamp_to_edge, sycl::addressing_mode::clamp};

  sycl::filtering_mode filtModes[2] = {sycl::filtering_mode::nearest,
                                       sycl::filtering_mode::linear};

  bool failed = false;

  for (auto addrMode : addrModes) {

    for (auto filtMode : filtModes) {

      if (normMode == sycl::coordinate_normalization_mode::unnormalized) {
        // These sampler combinations are not valid according to the SYCL spec
        if (addrMode == sycl::addressing_mode::repeat ||
            addrMode == sycl::addressing_mode::mirrored_repeat) {
          continue;
        }
      }
      // Skip using offset with address_mode of none. Will cause undefined
      // behaviour.
      if (addrMode == sycl::addressing_mode::none && offset != 0.0) {
        continue;
      }

      sycl::ext::oneapi::experimental::bindless_image_sampler samp(
          addrMode, normMode, filtMode);

#if defined(VERBOSE_LV2) || defined(VERBOSE_LV3)
      printTestInfo(samp, offset);
#endif

      // Tests using int data type currently disabled due to inconsistent
      // rounding behaviour against non-float types smaller then 32 bit.

      printTestName("Running 1D short\n");
      failed |=
          run_test<NDims, short, 1, sycl::image_channel_type::signed_int16,
                   sycl::image_channel_order::r, class short_1d1,
                   class short_1d2>(dims, localSize, offset, samp, seed);
      printTestName("Running 1D short2\n");
      failed |=
          run_test<NDims, short, 2, sycl::image_channel_type::signed_int16,
                   sycl::image_channel_order::rg, class short2_1d1,
                   class short2_1d2>(dims, localSize, offset, samp, seed);
      printTestName("Running 1D short4\n");
      failed |=
          run_test<NDims, short, 4, sycl::image_channel_type::signed_int16,
                   sycl::image_channel_order::rgba, class short4_1d1,
                   class short4_1d2>(dims, localSize, offset, samp, seed);

      printTestName("Running 1D unsigned short\n");
      failed |= run_test<
          NDims, unsigned short, 1, sycl::image_channel_type::unsigned_int16,
          sycl::image_channel_order::r, class ushort_1d1, class ushort_1d2>(
          dims, localSize, offset, samp, seed);
      printTestName("Running 1D unsigned short2\n");
      failed |= run_test<
          NDims, unsigned short, 2, sycl::image_channel_type::unsigned_int16,
          sycl::image_channel_order::rg, class ushort2_1d1, class ushort2_1d2>(
          dims, localSize, offset, samp, seed);
      printTestName("Running 1D unsigned short4\n");
      failed |=
          run_test<NDims, unsigned short, 4,
                   sycl::image_channel_type::unsigned_int16,
                   sycl::image_channel_order::rgba, class ushort4_1d1,
                   class ushort4_1d2>(dims, localSize, offset, samp, seed);

      printTestName("Running 1D char\n");
      failed |=
          run_test<NDims, signed char, 1, sycl::image_channel_type::signed_int8,
                   sycl::image_channel_order::r, class char_1d1,
                   class char_1d2>(dims, localSize, offset, samp, seed);
      printTestName("Running 1D char2\n");
      failed |=
          run_test<NDims, signed char, 2, sycl::image_channel_type::signed_int8,
                   sycl::image_channel_order::rg, class char2_1d1,
                   class char2_1d2>(dims, localSize, offset, samp, seed);
      printTestName("Running 1D char4\n");
      failed |=
          run_test<NDims, signed char, 4, sycl::image_channel_type::signed_int8,
                   sycl::image_channel_order::rgba, class char4_1d1,
                   class char4_1d2>(dims, localSize, offset, samp, seed);

      printTestName("Running 1D unsigned char\n");
      failed |= run_test<
          NDims, unsigned char, 1, sycl::image_channel_type::unsigned_int8,
          sycl::image_channel_order::r, class uchar_1d1, class uchar_1d2>(
          dims, localSize, offset, samp, seed);
      printTestName("Running 1D unsigned char2\n");
      failed |= run_test<
          NDims, unsigned char, 2, sycl::image_channel_type::unsigned_int8,
          sycl::image_channel_order::rg, class uchar2_1d1, class uchar2_1d2>(
          dims, localSize, offset, samp, seed);
      printTestName("Running 1D unsigned char4\n");
      failed |= run_test<
          NDims, unsigned char, 4, sycl::image_channel_type::unsigned_int8,
          sycl::image_channel_order::rgba, class uchar4_1d1, class uchar4_1d2>(
          dims, localSize, offset, samp, seed);

      printTestName("Running 1D float\n");
      failed |= run_test<NDims, float, 1, sycl::image_channel_type::fp32,
                         sycl::image_channel_order::r, class float_1d1,
                         class float_1d2>(dims, localSize, offset, samp, seed);
      printTestName("Running 1D float2\n");
      failed |= run_test<NDims, float, 2, sycl::image_channel_type::fp32,
                         sycl::image_channel_order::rg, class float2_1d1,
                         class float2_1d2>(dims, localSize, offset, samp, seed);
      printTestName("Running 1D float4\n");
      failed |= run_test<NDims, float, 4, sycl::image_channel_type::fp32,
                         sycl::image_channel_order::rgba, class float4_1d1,
                         class float4_1d2>(dims, localSize, offset, samp, seed);

      printTestName("Running 1D half\n");
      failed |= run_test<NDims, sycl::half, 1, sycl::image_channel_type::fp16,
                         sycl::image_channel_order::r, class half_1d1,
                         class half_1d2>(dims, localSize, offset, samp, seed);
      printTestName("Running 1D half2\n");
      failed |= run_test<NDims, sycl::half, 2, sycl::image_channel_type::fp16,
                         sycl::image_channel_order::rg, class half2_1d1,
                         class half2_1d2>(dims, localSize, offset, samp, seed);
      printTestName("Running 1D half4\n");
      failed |= run_test<NDims, sycl::half, 4, sycl::image_channel_type::fp16,
                         sycl::image_channel_order::rgba, class half4_1d1,
                         class half4_1d2>(dims, localSize, offset, samp, seed);

      printTestName("Running 1D float - dims: 1024, local: 512\n");
      failed |= run_test<NDims, float, 1, sycl::image_channel_type::fp32,
                         sycl::image_channel_order::r, class float_1d11,
                         class float_1d21>({1024}, {512}, offset, samp, seed);
      printTestName("Running 1D float4 - dims: 4096, local: 8\n");
      failed |= run_test<NDims, float, 4, sycl::image_channel_type::fp32,
                         sycl::image_channel_order::rgba, class float4_1d13,
                         class float4_1d23>({4096}, {8}, offset, samp, seed);
    }
  }

  return !failed;
}

template <int NDims>
bool run_offset(sycl::range<NDims> dims, sycl::range<NDims> localSize,
                double offset, int seed) {
  bool normPassed =
      run_tests<NDims>(dims, localSize, (offset / (double)dims[0]), seed,
                       sycl::coordinate_normalization_mode::normalized);
  bool nonormPassed =
      run_tests<NDims>(dims, localSize, offset, seed,
                       sycl::coordinate_normalization_mode::unnormalized);
  return normPassed && nonormPassed;
}

template <int NDims>
bool run_no_offset(sycl::range<NDims> dims, sycl::range<NDims> localSize,
                   int seed) {
  bool normPassed =
      run_tests<NDims>(dims, localSize, 0.0, seed,
                       sycl::coordinate_normalization_mode::normalized);
  bool nonormPassed =
      run_tests<NDims>(dims, localSize, 0.0, seed,
                       sycl::coordinate_normalization_mode::unnormalized);
  return normPassed && nonormPassed;
}

template <int NDims>
bool run_dim(sycl::range<NDims> dims, sycl::range<NDims> localSize,
             double offset, int seed) {
  bool offsetPassed = run_offset<NDims>(dims, localSize, offset, seed);
  bool noOffsetPassed = run_no_offset<NDims>(dims, localSize, seed);
  return offsetPassed && noOffsetPassed;
}

bool run_all(int seed) { return run_dim<1>({512}, {32}, 20, seed); }

int main() {

  unsigned int seed = 0;
  bool result = run_all(seed);

  if (result) {
    std::cout << "All tests passed!\n";
    return 0;
  }

  std::cerr << "An error has occured!\n";
  return 1;
}
