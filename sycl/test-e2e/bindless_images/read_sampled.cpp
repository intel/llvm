// REQUIRES: linux
// REQUIRES: cuda

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %t.out

// Print test names and pass status
// #define VERBOSE_LV1

// Same as above plus sampler, offset, margin of error, largest error found and
// results of one mismatch
// #define VERBOSE_LV2

// Same as above but all mismatches are printed
// #define VERBOSE_LV3

#include "bindless_helpers.hpp"
#include <cassert>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

namespace util {
static bool isNumberWithinPercentOfNumber(float firstN, float percent,
                                          float secondN, float &diff,
                                          float &percDiff) {
  // Get absolute difference of the two numbers
  diff = std::abs(firstN - secondN);
  // Get the percentage difference of the two numbers
  percDiff =
      100.0f * (std::abs(firstN - secondN) / (((firstN + secondN) / 2.0f)));

  // Check if perc difference is not greater then maximum allowed
  return percDiff <= percent;
}

// Return fractional part of argument
// Whole part is returned through wholeComp
static float fract(float x, float *wholeComp) {
  // This fmin operation is to prevent fract from returning 1.0.
  // Instead will return the largest possible floating-point number less
  // than 1.0
  float fractComp = std::fmin(x - std::floor(x), 0x1.fffffep-1f);
  *wholeComp = std::floor(x);
  return fractComp;
}

// Returns the two pixels to access plus the weight each of them have
static float getCommonLinearFractAndCoords(float coord, int *x0, int *x1) {
  float pixelCoord;

  // Subtract to align so that pixel center is 0.5 away from origin.
  coord = coord - 0.5;

  float weight = fract(coord, &pixelCoord);
  *x0 = static_cast<int>(std::floor(pixelCoord));
  *x1 = *x0 + 1;
  return weight;
}

// Linear sampling is the process of giving a weighted linear blend
// between the nearest adjacent pixels.
// When performing 1D linear sampling, we subtract 0.5 from the original
// coordinate to get the center-adjusted coordinate (as pixels are "centered"
// on the half-integers). For example, with original coord 3.2, we get a
// center-adjusted coord of 2.7. With 2.7, we have 70% of the pixel value will
// come from the pixel at coord 3 and 30% from the pixel value at coord 2

// The function accepts 4 pixel and 2 weight arguments for 2D linear sampling,
// but also supports 1D linear sampling by setting the unneeded arguments to 0
template <int NChannels, typename DType>
static sycl::vec<DType, NChannels>
linearOp(sycl::vec<DType, NChannels> pix1, sycl::vec<DType, NChannels> pix2,
         sycl::vec<DType, NChannels> pix3, sycl::vec<DType, NChannels> pix4,
         float weight1, float weight2) {

  sycl::vec<float, NChannels> weightArr1(weight1);
  sycl::vec<float, NChannels> weightArr2(weight2);
  sycl::vec<float, NChannels> one(1.0f);

  sycl::vec<float, NChannels> Ti0j0 = pix1.template convert<float>();
  sycl::vec<float, NChannels> Ti1j0 = pix2.template convert<float>();
  sycl::vec<float, NChannels> Ti0j1 = pix3.template convert<float>();
  sycl::vec<float, NChannels> Ti1j1 = pix4.template convert<float>();

  sycl::vec<float, NChannels> result;

  result = (((one - weightArr1) * (one - weightArr2) * Ti0j0 +
             weightArr1 * (one - weightArr2) * Ti1j0 +
             (one - weightArr1) * weightArr2 * Ti0j1 +
             weightArr1 * weightArr2 * Ti1j1));

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

// This prevents when at an integer junction, having three
// accesses to pixel at normalized location 0 and 1 instead of two which is
// correct.
static int integerJunctionAdjustment(float normCoord, int dimSize) {
  int oddShift = 0;
  // If not at image boundry and precisely on a pixel
  if (std::fmod(normCoord, 1) != 0.0 &&
      std::fmod(normCoord * dimSize, 1) == 0.0) {
    // Set oddShift to be one when the integral part of the normalized
    // coords is odd.
    // Otherwise set to zero.
    oddShift = std::abs(static_cast<int>(std::fmod(std::floor(normCoord), 2)));
  }
  return oddShift;
}

// Wrap linear sampling coords to valid range
static int repeatWrap(int i, int dimSize) {
  if (i < 0) {
    i = dimSize + i;
  }
  if (i > dimSize - 1) {
    i = i - dimSize;
  }
  return i;
}

static void printTestInfo(syclexp::bindless_image_sampler &samp, float offset) {

  sycl::addressing_mode SampAddrMode = samp.addressing[0];
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

// Out of range coords return a border color
// The border color happens to be all zeros
template <typename VecType>
static VecType clampNearest(sycl::vec<float, 2> coords,
                            sycl::range<2> globalSize,
                            std::vector<VecType> &inputImage) {
  float coordX = coords[0];
  float coordY = coords[1];
  int width = globalSize[0];
  int height = globalSize[1];

  // Due to pixel centers being 0.5 away from origin and because
  // 0.5 is *not* subtracted here, rounding down gives the same results as
  // rounding to nearest number if 0.5 is subtracted to account
  // for pixel center
  int coordXInt = static_cast<int>(std::floor(coordX));
  int coordYInt = static_cast<int>(std::floor(coordY));

  // Clamp sampling according to the SYCL spec returns a border color.
  // The border color is all zeros.
  // There does not appear to be any way for the user to set the border color
  if (coordXInt > width - 1) {
    return VecType{0};
  }
  if (coordXInt < 0) {
    return VecType{0};
  }
  if (coordYInt > height - 1) {
    return VecType{0};
  }
  if (coordYInt < 0) {
    return VecType{0};
  }

  return inputImage[coordX + (width * coordY)];
}

// Out of range coords are clamped to the extent.
static int clampToEdgeNearestCoord(float coord, int dimSize) {
  // Due to pixel centers being 0.5 away from origin and because
  // 0.5 is *not* subtracted here, rounding down gives the same results as
  // rounding to nearest number if 0.5 is subtracted to account
  // for pixel center
  int coordInt = static_cast<int>(std::floor(coord));

  // Clamp to extent
  coordInt = std::clamp(coordInt, 0, dimSize - 1);

  return coordInt;
}

// Out of range coords are clamped to the extent.
template <typename VecType>
static VecType clampToEdgeNearest(sycl::vec<float, 2> coords,
                                  sycl::range<2> globalSize,
                                  std::vector<VecType> &inputImage) {
  int width = globalSize[0];

  int coordXInt =
      clampToEdgeNearestCoord(coords[0], static_cast<int>(globalSize[0]));
  int coordYInt =
      clampToEdgeNearestCoord(coords[1], static_cast<int>(globalSize[1]));

  return inputImage[coordXInt + (width * coordYInt)];
}

// Out of range coords are wrapped to the valid range.
static int repeatNearestCoord(float coord, int dimSize) {
  // Convert unnormalized input coord to normalized format
  float normCoord = coord / dimSize;

  // Keep only the fractional component of the number and unnormalize.
  float fractComp = (normCoord - std::floor(normCoord));

  // Unnormalize fractComp
  float unnorm = fractComp * dimSize;

  // Due to pixel centers being 0.5 away from origin and because
  // 0.5 is *not* subtracted here, rounding down gives the same results as
  // rounding to nearest number if 0.5 is subtracted to account
  // for pixel center
  int coordInt = static_cast<int>(std::floor(unnorm));

  // Handle negative coords
  if (coordInt < 0) {
    coordInt = dimSize + coordInt;
  }

  return coordInt;
}

// Out of range coords are wrapped to the valid range.
template <typename VecType>
static VecType repeatNearest(sycl::vec<float, 2> coords,
                             sycl::range<2> globalSize,
                             std::vector<VecType> &inputImage) {
  int width = globalSize[0];

  int coordXInt =
      util::repeatNearestCoord(coords[0], static_cast<int>(globalSize[0]));
  int coordYInt =
      util::repeatNearestCoord(coords[1], static_cast<int>(globalSize[1]));

  return inputImage[coordXInt + (width * coordYInt)];
}

// Out of range coordinates are flipped at every integer junction
static int mirroredRepeatNearestCoord(float coord, int dimSize) {

  // Convert unnormalized input coord to normalized format
  float normCoord = coord / dimSize;

  // Round to nearest multiple of two.
  // e.g.
  // normCoord == 0.3  -> result = 0
  // normCoord == 1.3  -> result = 2
  // normCoord == 2.4  -> result = 2
  // normCoord == 3.42 -> result = 4
  float nearestMulOfTwo = 2.0f * std::rint(0.5f * normCoord);

  // Subtract nearestMulOfTwo from normCoordX.
  // Gives the normalized form of the coord to use.
  // With normCoord=1.3, norm is set to 0.7
  // With normCoord=2.4, norm is set to 0.4
  float norm = std::abs(normCoord - nearestMulOfTwo);

  // Unnormalize norm
  float unnorm = norm * dimSize;

  // Round down and cast to int
  int coordInt = static_cast<int>(std::floor(unnorm));

  // Constrain to valid range
  coordInt = std::min(coordInt, dimSize - 1);

  // This prevents when at an integer junction, having three
  // accesses to pixel at normalized location 0 and 1 instead of two which is
  // correct.
  coordInt -= integerJunctionAdjustment(normCoord, dimSize);

  return coordInt;
}

// Out of range coordinates are flipped at every integer junction
template <typename VecType>
static VecType mirroredRepeatNearest(sycl::vec<float, 2> coords,
                                     sycl::range<2> globalSize,
                                     std::vector<VecType> &inputImage) {
  int width = globalSize[0];

  int coordXInt = util::mirroredRepeatNearestCoord(
      coords[0], static_cast<int>(globalSize[0]));
  int coordYInt = util::mirroredRepeatNearestCoord(
      coords[1], static_cast<int>(globalSize[1]));

  return inputImage[coordXInt + (width * coordYInt)];
}

template <typename VecType>
static VecType noneNearest(sycl::vec<float, 2> coords,
                           sycl::range<2> globalSize,
                           std::vector<VecType> &inputImage) {
  int intCoordX = static_cast<int>(std::floor(coords[0]));
  int intCoordY = static_cast<int>(std::floor(coords[1]));
  int width = globalSize[0];

  return inputImage[intCoordX + (width * intCoordY)];
}

// Clamp sampling according to the SYCL spec returns a border color.
// The border color is all zeros.
// There does not appear to be any way for the user to set the border color.
template <typename VecType>
static VecType clampLinearCheckBounds(int i, int j, int width, int height,
                                      std::vector<VecType> &inputImage) {
  if (i < 0 || i > width - 1 || j < 0 || j > height - 1) {
    return VecType(0);
  }
  return inputImage[i + (width * j)];
}

struct InterpolRes {
  int x0;
  int x1;
  float weight;
  InterpolRes(int tempX0, int tempX1, float tempWeight)
      : x0(tempX0), x1(tempX1), weight(tempWeight) {}
};

// Out of range coords return a border color
// The border color is all zeros
template <typename DType, int NChannels>
static sycl::vec<DType, NChannels>
clampLinear(sycl::vec<float, 2> coords, sycl::range<2> globalSize,
            std::vector<sycl::vec<DType, NChannels>> &inputImage) {
  using VecType = sycl::vec<DType, NChannels>;

  float coordX = coords[0];
  float coordY = coords[1];
  int width = globalSize[0];
  int height = globalSize[1];

  // Get coords for linear sampling
  int i0, i1;
  float weightX = util::getCommonLinearFractAndCoords(coordX, &i0, &i1);

  int j0 = 0, j1 = 0;
  // If height is not one, run as normal.
  // Otherwise, keep weightY set to 0.
  float weightY =
      height == 1 ? 0 : util::getCommonLinearFractAndCoords(coordY, &j0, &j1);

  // Clamp sampling according to the SYCL spec returns a border color.
  // The border color is all zeros.
  // There does not appear to be any way for the user to set the border color.
  VecType pix1 =
      clampLinearCheckBounds<VecType>(i0, j0, width, height, inputImage);
  VecType pix2 =
      clampLinearCheckBounds<VecType>(i1, j0, width, height, inputImage);
  VecType pix3 =
      clampLinearCheckBounds<VecType>(i0, j1, width, height, inputImage);
  VecType pix4 =
      clampLinearCheckBounds<VecType>(i1, j1, width, height, inputImage);

  // Perform linear sampling
  return util::linearOp<NChannels, DType>(pix1, pix2, pix3, pix4, weightX,
                                          weightY);
}

// Out of range coords are clamped to the extent.
template <typename DType, int NChannels>
static sycl::vec<DType, NChannels>
clampToEdgeLinear(sycl::vec<float, 2> coords, sycl::range<2> globalSize,
                  std::vector<sycl::vec<DType, NChannels>> &inputImage) {
  using VecType = sycl::vec<DType, NChannels>;

  float coordX = coords[0];
  float coordY = coords[1];
  int width = globalSize[0];
  int height = globalSize[1];

  // Get coords for linear sampling
  int i0, i1;
  float weightX = util::getCommonLinearFractAndCoords(coordX, &i0, &i1);

  int j0 = 0, j1 = 0;
  // If height is not one, run as normal.
  // Otherwise, keep weightY set to 0.
  float weightY =
      height == 1 ? 0 : util::getCommonLinearFractAndCoords(coordY, &j0, &j1);

  // Clamp to extent
  i0 = std::clamp(i0, 0, width - 1);
  i1 = std::clamp(i1, 0, width - 1);
  j0 = std::clamp(j0, 0, height - 1);
  j1 = std::clamp(j1, 0, height - 1);

  VecType pix1 = inputImage[i0 + (width * j0)];
  VecType pix2 = inputImage[i1 + (width * j0)];
  VecType pix3 = inputImage[i0 + (width * j1)];
  VecType pix4 = inputImage[i1 + (width * j1)];

  // Perform linear sampling
  return util::linearOp<NChannels, DType>(pix1, pix2, pix3, pix4, weightX,
                                          weightY);
}

// Out of range coords return a border color
// The border color is all zeros
static InterpolRes repeatLinearCoord(float coord, int dimSize) {

  // Convert unnormalized input coord to normalized format
  float normCoord = coord / dimSize;

  float unnorm = (normCoord - static_cast<int>(normCoord)) * dimSize;

  // Get coords for linear sampling
  int x0, x1;
  float weight = getCommonLinearFractAndCoords(unnorm, &x0, &x1);

  return InterpolRes(x0, x1, weight);
}

// Out of range coords are wrapped to the valid range
template <typename DType, int NChannels>
static sycl::vec<DType, NChannels>
repeatLinear(sycl::vec<float, 2> coords, sycl::range<2> globalSize,
             std::vector<sycl::vec<DType, NChannels>> &inputImage) {
  using VecType = sycl::vec<DType, NChannels>;

  float coordX = coords[0];
  float coordY = coords[1];
  int width = globalSize[0];
  int height = globalSize[1];

  util::InterpolRes resX = util::repeatLinearCoord(coordX, width);
  // If height is not one, run as normal.
  // Otherwise, set resY to all zeros.
  util::InterpolRes resY = height == 1
                               ? InterpolRes(0, 0, 0)
                               : util::repeatLinearCoord(coordY, height);

  int i0 = resX.x0, i1 = resX.x1;
  int j0 = resY.x0, j1 = resY.x1;

  float weightX = resX.weight, weightY = resY.weight;

  // Wrap linear sampling coords to valid range
  i0 = util::repeatWrap(i0, width);
  i1 = util::repeatWrap(i1, width);
  j0 = util::repeatWrap(j0, height);
  j1 = util::repeatWrap(j1, height);

  VecType pix1 = inputImage[i0 + (width * j0)];
  VecType pix2 = inputImage[i1 + (width * j0)];
  VecType pix3 = inputImage[i0 + (width * j1)];
  VecType pix4 = inputImage[i1 + (width * j1)];

  // Perform linear sampling
  return util::linearOp<NChannels, DType>(pix1, pix2, pix3, pix4, weightX,
                                          weightY);
}

// Out of range coordinates are flipped at every integer junction
static InterpolRes mirroredRepeatLinearCoord(float coord, int dimSize) {

  // Convert unnormalized input coord to normalized format
  float normCoord = coord / dimSize;

  // Round to nearest multiple of two.
  // e.g.
  // normCoordX == 0.3  -> result = 0
  // normCoordX == 1.3  -> result = 2
  // normCoordX == 2.4  -> result = 2
  // normCoordX == 3.42 -> result = 4
  float nearestMulOfTwo = 2.0f * std::rint(0.5f * normCoord);

  // Subtract nearestMulOfTwo from normCoordX.
  // Gives the normalized form of the coord to use.
  // With normCoordX=1.3, norm is set to 0.7
  // With normCoordX=2.4, norm is set to 0.4
  float norm = std::abs(normCoord - nearestMulOfTwo);

  // Unnormalize norm
  float unnorm = norm * dimSize;

  // Get coords for linear sampling
  int x0, x1;
  float weight = getCommonLinearFractAndCoords(unnorm, &x0, &x1);

  return InterpolRes(x0, x1, weight);
}

// Out of range coordinates are flipped at every integer junction
template <typename DType, int NChannels>
static sycl::vec<DType, NChannels>
mirroredRepeatLinear(sycl::vec<float, 2> coords, sycl::range<2> globalSize,
                     std::vector<sycl::vec<DType, NChannels>> &inputImage) {
  using VecType = sycl::vec<DType, NChannels>;

  float coordX = coords[0];
  float coordY = coords[1];
  int width = globalSize[0];
  int height = globalSize[1];

  util::InterpolRes resX = util::mirroredRepeatLinearCoord(coordX, width);
  // If height is not one, run as normal.
  // Otherwise, set resY to all zeros.
  util::InterpolRes resY =
      height == 1 ? InterpolRes(0, 0, 0)
                  : util::mirroredRepeatLinearCoord(coordY, height);

  int i0 = resX.x0, i1 = resX.x1;
  int j0 = resY.x0, j1 = resY.x1;

  float weightX = resX.weight, weightY = resY.weight;

  // getCommonLinear sometimes returns numbers out of bounds.
  // Handle this by wrapping to boundary.
  i0 = std::max(i0, 0);
  i1 = std::min(i1, width - 1);
  j0 = std::max(j0, 0);
  j1 = std::min(j1, height - 1);

  VecType pix1 = inputImage[i0 + (width * j0)];
  VecType pix2 = inputImage[i1 + (width * j0)];
  VecType pix3 = inputImage[i0 + (width * j1)];
  VecType pix4 = inputImage[i1 + (width * j1)];

  // Perform linear sampling
  return util::linearOp<NChannels, DType>(pix1, pix2, pix3, pix4, weightX,
                                          weightY);
}

// Some vector sizes here are hardcoded because the sampling functions are
// designed to only accept vecs of that size.
template <int NDims, typename DType, int NChannels>
static sycl::vec<DType, NChannels>
read(sycl::range<2> globalSize, sycl::vec<float, 2> coords, float offset,
     syclexp::bindless_image_sampler &samp,
     std::vector<sycl::vec<DType, NChannels>> &inputImage) {
  using VecType = sycl::vec<DType, NChannels>;

  // Add offset to coords
  for (int i = 0; i < NDims; i++) {
    coords[i] = coords[i] + offset;
  }

  // Ensure that coords always contain unnormalized coords
  sycl::coordinate_normalization_mode SampNormMode = samp.coordinate;
  if (SampNormMode == sycl::coordinate_normalization_mode::normalized) {
    // Unnormalize
    for (int i = 0; i < NDims; i++) {
      coords[i] = coords[i] * globalSize[i];
    }
  }

  sycl::filtering_mode SampFiltMode = samp.filtering;
  if (SampFiltMode == sycl::filtering_mode::nearest) {

    sycl::addressing_mode SampAddrMode = samp.addressing[0];
    if (SampAddrMode == sycl::addressing_mode::clamp) {
      return util::clampNearest<VecType>(coords, globalSize, inputImage);
    }

    if (SampAddrMode == sycl::addressing_mode::clamp_to_edge) {
      return util::clampToEdgeNearest<VecType>(coords, globalSize, inputImage);
    }

    if (SampAddrMode == sycl::addressing_mode::repeat) {
      if (SampNormMode == sycl::coordinate_normalization_mode::unnormalized) {
        assert(false &&
               "Repeat addressing mode must be used with normalized coords");
      }
      return util::repeatNearest<VecType>(coords, globalSize, inputImage);
    }

    if (SampAddrMode == sycl::addressing_mode::mirrored_repeat) {
      if (SampNormMode == sycl::coordinate_normalization_mode::unnormalized) {
        assert(false && "Mirrored repeat addressing mode must be used with "
                        "normalized coords");
      }
      return util::mirroredRepeatNearest<VecType>(coords, globalSize,
                                                  inputImage);
    }

    if (SampAddrMode == sycl::addressing_mode::none) {
      // Ensure no access out of bounds when addressing_mode is none
      // due to that being undefined behaviour.
      bool outOfBounds = false;
      for (int i = 0; i < NDims; i++) {
        int intCoord = static_cast<int>(std::floor(coords[i]));
        outOfBounds =
            outOfBounds || (intCoord < 0 || intCoord >= globalSize[i]);
      }
      if (outOfBounds) {
        assert(false && "Accessed out of bounds with addressing mode none! "
                        "Undefined Behaviour!");
      }
      return util::noneNearest(coords, globalSize, inputImage);
    }

  } else { // linear
    sycl::addressing_mode SampAddrMode = samp.addressing[0];
    if (SampAddrMode == sycl::addressing_mode::clamp) {
      return util::clampLinear<DType, NChannels>(coords, globalSize,
                                                 inputImage);
    }
    if (SampAddrMode == sycl::addressing_mode::clamp_to_edge) {
      return util::clampToEdgeLinear<DType, NChannels>(coords, globalSize,
                                                       inputImage);
    }
    if (SampAddrMode == sycl::addressing_mode::repeat) {
      if (SampNormMode == sycl::coordinate_normalization_mode::unnormalized) {
        assert(false &&
               "Repeat addressing mode must be used with normalized coords");
      }
      return util::repeatLinear<DType, NChannels>(coords, globalSize,
                                                  inputImage);
    }
    if (SampAddrMode == sycl::addressing_mode::mirrored_repeat) {
      if (SampNormMode == sycl::coordinate_normalization_mode::unnormalized) {
        assert(false && "Mirrored repeat addressing mode must be used with "
                        "normalized coords");
      }
      return util::mirroredRepeatLinear<DType, NChannels>(coords, globalSize,
                                                          inputImage);
    }
    if (SampAddrMode == sycl::addressing_mode::none) {
      // Ensure no access out of bounds when addressing_mode is none
      // due to that being undefined behaviour.
      bool outOfBounds = false;
      for (int i = 0; i < NDims; i++) {
        outOfBounds =
            outOfBounds || (coords[i] < 0 || coords[i] >= globalSize[i]);
      }
      if (outOfBounds) {
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
static void
runNDimTestHost(sycl::range<NDims> globalSize, float offset,
                syclexp::bindless_image_sampler &samp,
                std::vector<sycl::vec<DType, NChannels>> &inputImage,
                std::vector<sycl::vec<DType, NChannels>> &output) {

  using VecType = sycl::vec<DType, NChannels>;
  bool isNorm =
      (samp.coordinate == sycl::coordinate_normalization_mode::normalized);

  sycl::vec<float, 2> coords;

  sycl::range<2> globalSizeTwoComp;

  for (int i = 0; i < NDims; i++) {
    globalSizeTwoComp[i] = globalSize[i];
  }

  for (int i = 0; i < 2 - NDims; i++) {
    globalSizeTwoComp[NDims - i] = 1;
  }

  for (int i = 0; i < globalSizeTwoComp[0]; i++) {
    for (int j = 0; j < globalSizeTwoComp[1]; j++) {
      if (isNorm) {
        coords[0] = (float)i / (float)globalSizeTwoComp[0];
        coords[1] = (float)j / (float)globalSizeTwoComp[1];

      } else {
        coords[0] = i;
        coords[1] = j;
      }

      VecType result = read<NDims, DType, NChannels>(globalSizeTwoComp, coords,
                                                     offset, samp, inputImage);
      output[i + (globalSize[0] * j)] = result;
    }
  }
}

// parallel_for ND bindless normalized
template <int NDims, typename DType, int NChannels, typename KernelName>
static void
runNDimTestDevice(sycl::queue &q, sycl::range<NDims> globalSize,
                  sycl::range<NDims> localSize, float offset,
                  syclexp::bindless_image_sampler &samp,
                  syclexp::sampled_image_handle inputImage,
                  sycl::buffer<sycl::vec<DType, NChannels>, NDims> &output,
                  sycl::range<NDims> bufSize) {

  using VecType = sycl::vec<DType, NChannels>;
  bool isNorm =
      (samp.coordinate == sycl::coordinate_normalization_mode::normalized);
  try {
    q.submit([&](sycl::handler &cgh) {
      auto outAcc =
          output.template get_access<sycl::access_mode::write>(cgh, bufSize);
      cgh.parallel_for<KernelName>(
          sycl::nd_range<NDims>{globalSize, localSize},
          [=](sycl::nd_item<NDims> it) {
            sycl::id<NDims> accessorCoords;
            sycl::float2 coords;

            if (isNorm) {
              for (int i = 0; i < NDims; i++) {
                coords[i] =
                    ((float)it.get_global_id(i) / (float)globalSize[i]) +
                    offset;
              }

            } else {
              for (int i = 0; i < NDims; i++) {
                coords[i] = (float)it.get_global_id(i) + offset;
              }
            }
            for (int i = 0; i < NDims; i++) {
              // Accessor coords are to be inverted.
              accessorCoords[i] = it.get_global_id(NDims - i - 1);
            }

            VecType px1 = syclexp::sample_image<VecType>(inputImage, coords);

            outAcc[accessorCoords] = px1;
          });
    });
  } catch (sycl::exception e) {
    std::cerr << "\tKernel submission failed! " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "\tKernel submission failed!" << std::endl;
  }
}

template <int NDims, typename DType, int NChannels,
          sycl::image_channel_type CType, sycl::image_channel_order COrder,
          typename KernelName>
static bool runTest(sycl::range<NDims> dims, sycl::range<NDims> localSize,
                    float offset, syclexp::bindless_image_sampler &samp,
                    unsigned int seed = 0) {
  using VecType = sycl::vec<DType, NChannels>;

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  size_t numElems = dims[0];
  if constexpr (NDims == 2) {
    numElems *= dims[1];
  }

  std::vector<VecType> input(numElems);
  std::vector<VecType> expected(numElems);
  std::vector<VecType> actual(numElems);

  std::srand(seed);
  bindless_helpers::fill_rand(input, seed);

  {
    sycl::range<NDims> globalSize = dims;
    runNDimTestHost<NDims, DType, NChannels>(globalSize, offset, samp, input,
                                             expected);
  }

  try {

    syclexp::image_descriptor desc(dims, COrder, CType);

    syclexp::image_mem imgMem(desc, q);

    auto img_input = syclexp::create_image(imgMem, samp, desc, q);

    q.ext_oneapi_copy(input.data(), imgMem.get_handle(), desc);
    q.wait_and_throw();

    {
      sycl::range<NDims> bufSize = dims;
      sycl::range<NDims> globalSize = dims;
      sycl::buffer<VecType, NDims> outBuf((VecType *)actual.data(), bufSize);
      q.wait_and_throw();
      runNDimTestDevice<NDims, DType, NChannels, KernelName>(
          q, globalSize, localSize, offset, samp, img_input, outBuf, bufSize);
      q.wait_and_throw();
    }

    // Cleanup
    syclexp::destroy_image_handle(img_input, q);

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
  for (int i = 0; i < numElems; i++) {
    for (int j = 0; j < NChannels; ++j) {
      bool mismatch = false;
      if (actual[i][j] != expected[i][j]) {
        // Nvidia GPUs have a 0.4%~ margin of error due to only using 8 bits to
        // represent values between 1-0 for weights during linear interpolation.
        float diff, percDiff;
        if (!util::isNumberWithinPercentOfNumber(
                actual[i][j], deviation, expected[i][j], diff, percDiff)) {
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
    std::cout << "\tCorrect output!\n";
  } else {
    std::cout << "\tIncorrect output!\n";
  }
#endif

  return !validated;
}

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
}; // namespace util

template <int NDims, typename = std::enable_if_t<NDims == 1>>
bool runTests(sycl::range<1> dims, sycl::range<1> localSize, float offset,
              int seed, sycl::coordinate_normalization_mode normMode) {

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

      syclexp::bindless_image_sampler samp(addrMode, normMode, filtMode);

#if defined(VERBOSE_LV2) || defined(VERBOSE_LV3)
      util::printTestInfo(samp, offset);
#endif

      util::printTestName<NDims>("Running 1D short", dims, localSize);
      failed |=
          util::runTest<NDims, short, 1, sycl::image_channel_type::signed_int16,
                        sycl::image_channel_order::r, class short_1d>(
              dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 1D short2", dims, localSize);
      failed |=
          util::runTest<NDims, short, 2, sycl::image_channel_type::signed_int16,
                        sycl::image_channel_order::rg, class short2_1d>(
              dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 1D short4", dims, localSize);
      failed |=
          util::runTest<NDims, short, 4, sycl::image_channel_type::signed_int16,
                        sycl::image_channel_order::rgba, class short4_1d>(
              dims, localSize, offset, samp, seed);

      util::printTestName<NDims>("Running 1D unsigned short", dims, localSize);
      failed |= util::runTest<NDims, unsigned short, 1,
                              sycl::image_channel_type::unsigned_int16,
                              sycl::image_channel_order::r, class ushort_1d>(
          dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 1D unsigned short2", dims, localSize);
      failed |= util::runTest<NDims, unsigned short, 2,
                              sycl::image_channel_type::unsigned_int16,
                              sycl::image_channel_order::rg, class ushort2_1d>(
          dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 1D unsigned short4", dims, localSize);
      failed |=
          util::runTest<NDims, unsigned short, 4,
                        sycl::image_channel_type::unsigned_int16,
                        sycl::image_channel_order::rgba, class ushort4_1d>(
              dims, localSize, offset, samp, seed);

      util::printTestName<NDims>("Running 1D char", dims, localSize);
      failed |= util::runTest<NDims, signed char, 1,
                              sycl::image_channel_type::signed_int8,
                              sycl::image_channel_order::r, class char_1d>(
          dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 1D char2", dims, localSize);
      failed |= util::runTest<NDims, signed char, 2,
                              sycl::image_channel_type::signed_int8,
                              sycl::image_channel_order::rg, class char2_1d>(
          dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 1D char4", dims, localSize);
      failed |= util::runTest<NDims, signed char, 4,
                              sycl::image_channel_type::signed_int8,
                              sycl::image_channel_order::rgba, class char4_1d>(
          dims, localSize, offset, samp, seed);

      util::printTestName<NDims>("Running 1D unsigned char", dims, localSize);
      failed |= util::runTest<NDims, unsigned char, 1,
                              sycl::image_channel_type::unsigned_int8,
                              sycl::image_channel_order::r, class uchar_1d>(
          dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 1D unsigned char2", dims, localSize);
      failed |= util::runTest<NDims, unsigned char, 2,
                              sycl::image_channel_type::unsigned_int8,
                              sycl::image_channel_order::rg, class uchar2_1d>(
          dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 1D unsigned char4", dims, localSize);
      failed |= util::runTest<NDims, unsigned char, 4,
                              sycl::image_channel_type::unsigned_int8,
                              sycl::image_channel_order::rgba, class uchar4_1d>(
          dims, localSize, offset, samp, seed);

      util::printTestName<NDims>("Running 1D float", dims, localSize);
      failed |= util::runTest<NDims, float, 1, sycl::image_channel_type::fp32,
                              sycl::image_channel_order::r, class float_1d>(
          dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 1D float2", dims, localSize);
      failed |= util::runTest<NDims, float, 2, sycl::image_channel_type::fp32,
                              sycl::image_channel_order::rg, class float2_1d>(
          dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 1D float4", dims, localSize);
      failed |= util::runTest<NDims, float, 4, sycl::image_channel_type::fp32,
                              sycl::image_channel_order::rgba, class float4_1d>(
          dims, localSize, offset, samp, seed);

      util::printTestName<NDims>("Running 1D half", dims, localSize);
      failed |=
          util::runTest<NDims, sycl::half, 1, sycl::image_channel_type::fp16,
                        sycl::image_channel_order::r, class half_1d>(
              dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 1D half2", dims, localSize);
      failed |=
          util::runTest<NDims, sycl::half, 2, sycl::image_channel_type::fp16,
                        sycl::image_channel_order::rg, class half2_1d>(
              dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 1D half4", dims, localSize);
      failed |=
          util::runTest<NDims, sycl::half, 4, sycl::image_channel_type::fp16,
                        sycl::image_channel_order::rgba, class half4_1d>(
              dims, localSize, offset, samp, seed);

      util::printTestName<NDims>("Running 1D float", {512}, {32});
      failed |= util::runTest<NDims, float, 1, sycl::image_channel_type::fp32,
                              sycl::image_channel_order::r, class float_1d1>(
          {512}, {32}, offset, samp, seed);
      util::printTestName<NDims>("Running 1D float4", {512}, {8});
      failed |=
          util::runTest<NDims, float, 4, sycl::image_channel_type::fp32,
                        sycl::image_channel_order::rgba, class float4_1d2>(
              {512}, {8}, offset, samp, seed);
    }
  }

  return !failed;
}

template <int NDims, typename = std::enable_if_t<NDims == 2>>
bool runTests(sycl::range<2> dims, sycl::range<2> localSize, float offset,
              int seed, sycl::coordinate_normalization_mode normMode) {

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

      syclexp::bindless_image_sampler samp(addrMode, normMode, filtMode);

#if defined(VERBOSE_LV2) || defined(VERBOSE_LV3)
      util::printTestInfo(samp, offset);
#endif

      util::printTestName<NDims>("Running 2D short", dims, localSize);
      failed |=
          util::runTest<NDims, short, 1, sycl::image_channel_type::signed_int16,
                        sycl::image_channel_order::r, class short_2d>(
              dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 2D short2", dims, localSize);
      failed |=
          util::runTest<NDims, short, 2, sycl::image_channel_type::signed_int16,
                        sycl::image_channel_order::rg, class short2_2d>(
              dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 2D short4", dims, localSize);
      failed |=
          util::runTest<NDims, short, 4, sycl::image_channel_type::signed_int16,
                        sycl::image_channel_order::rgba, class short4_2d>(
              dims, localSize, offset, samp, seed);

      util::printTestName<NDims>("Running 2D unsigned short", dims, localSize);
      failed |= util::runTest<NDims, unsigned short, 1,
                              sycl::image_channel_type::unsigned_int16,
                              sycl::image_channel_order::r, class ushort_2d>(
          dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 2D unsigned short2", dims, localSize);
      failed |= util::runTest<NDims, unsigned short, 2,
                              sycl::image_channel_type::unsigned_int16,
                              sycl::image_channel_order::rg, class ushort2_2d>(
          dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 2D unsigned short4", dims, localSize);
      failed |=
          util::runTest<NDims, unsigned short, 4,
                        sycl::image_channel_type::unsigned_int16,
                        sycl::image_channel_order::rgba, class ushort4_2d>(
              dims, localSize, offset, samp, seed);

      util::printTestName<NDims>("Running 2D char", dims, localSize);
      failed |= util::runTest<NDims, signed char, 1,
                              sycl::image_channel_type::signed_int8,
                              sycl::image_channel_order::r, class char_2d>(
          dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 2D char2", dims, localSize);
      failed |= util::runTest<NDims, signed char, 2,
                              sycl::image_channel_type::signed_int8,
                              sycl::image_channel_order::rg, class char2_2d>(
          dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 2D char4", dims, localSize);
      failed |= util::runTest<NDims, signed char, 4,
                              sycl::image_channel_type::signed_int8,
                              sycl::image_channel_order::rgba, class char4_2d>(
          dims, localSize, offset, samp, seed);

      util::printTestName<NDims>("Running 2D unsigned char", dims, localSize);
      failed |= util::runTest<NDims, unsigned char, 1,
                              sycl::image_channel_type::unsigned_int8,
                              sycl::image_channel_order::r, class uchar_2d>(
          dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 2D unsigned char2", dims, localSize);
      failed |= util::runTest<NDims, unsigned char, 2,
                              sycl::image_channel_type::unsigned_int8,
                              sycl::image_channel_order::rg, class uchar2_2d>(
          dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 2D unsigned char4", dims, localSize);
      failed |= util::runTest<NDims, unsigned char, 4,
                              sycl::image_channel_type::unsigned_int8,
                              sycl::image_channel_order::rgba, class uchar4_2d>(
          dims, localSize, offset, samp, seed);

      util::printTestName<NDims>("Running 2D float", dims, localSize);
      failed |= util::runTest<NDims, float, 1, sycl::image_channel_type::fp32,
                              sycl::image_channel_order::r, class float_2d>(
          dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 2D float2", dims, localSize);
      failed |= util::runTest<NDims, float, 2, sycl::image_channel_type::fp32,
                              sycl::image_channel_order::rg, class float2_2d>(
          dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 2D float4", dims, localSize);
      failed |= util::runTest<NDims, float, 4, sycl::image_channel_type::fp32,
                              sycl::image_channel_order::rgba, class float4_2d>(
          dims, localSize, offset, samp, seed);

      util::printTestName<NDims>("Running 2D half", dims, localSize);
      failed |=
          util::runTest<NDims, sycl::half, 1, sycl::image_channel_type::fp16,
                        sycl::image_channel_order::r, class half_2d>(
              dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 2D half2", dims, localSize);
      failed |=
          util::runTest<NDims, sycl::half, 2, sycl::image_channel_type::fp16,
                        sycl::image_channel_order::rg, class half2_2d>(
              dims, localSize, offset, samp, seed);
      util::printTestName<NDims>("Running 2D half4", dims, localSize);
      failed |=
          util::runTest<NDims, sycl::half, 4, sycl::image_channel_type::fp16,
                        sycl::image_channel_order::rgba, class half4_2d>(
              dims, localSize, offset, samp, seed);

      util::printTestName<NDims>("Running 2D float", {512, 512}, {32, 32});
      failed |= util::runTest<NDims, float, 1, sycl::image_channel_type::fp32,
                              sycl::image_channel_order::r, class float_2d1>(
          {512, 512}, {32, 32}, offset, samp, seed);
      util::printTestName<NDims>("Running 2D float4", {512, 512}, {8, 8});
      failed |=
          util::runTest<NDims, float, 4, sycl::image_channel_type::fp32,
                        sycl::image_channel_order::rgba, class float4_2d2>(
              {512, 512}, {8, 8}, offset, samp, seed);
    }
  }

  return !failed;
}

template <int NDims>
bool runOffset(sycl::range<NDims> dims, sycl::range<NDims> localSize,
               float offset, int seed) {
  bool normPassed = true;
  bool noNormPassed = true;
  normPassed = normPassed &&
               runTests<NDims>(dims, localSize, (offset / (float)dims[0]), seed,
                               sycl::coordinate_normalization_mode::normalized);
  noNormPassed =
      noNormPassed &&
      runTests<NDims>(dims, localSize, offset, seed,
                      sycl::coordinate_normalization_mode::unnormalized);
  return normPassed && noNormPassed;
}

template <int NDims>
bool runNoOffset(sycl::range<NDims> dims, sycl::range<NDims> localSize,
                 int seed) {
  bool normPassed = true;
  bool noNormPassed = true;
  normPassed = normPassed &&
               runTests<NDims>(dims, localSize, 0.0, seed,
                               sycl::coordinate_normalization_mode::normalized);
  noNormPassed =
      noNormPassed &&
      runTests<NDims>(dims, localSize, 0.0, seed,
                      sycl::coordinate_normalization_mode::unnormalized);
  return normPassed && noNormPassed;
}

template <int NDims>
bool runAll(sycl::range<NDims> dims, sycl::range<NDims> localSize, float offset,
            int seed) {
  bool offsetPassed = true;
  bool noOffsetPassed = true;
  offsetPassed =
      offsetPassed && runOffset<NDims>(dims, localSize, offset, seed);
  noOffsetPassed = noOffsetPassed && runNoOffset<NDims>(dims, localSize, seed);
  return offsetPassed && noOffsetPassed;
}

int main() {

  unsigned int seed = 0;
  std::cout << "Running 1D Sampled Image Tests!\n";
  bool result1D = runAll<1>({256}, {32}, 20, seed);
  std::cout << "Running 2D Sampled Image Tests!\n";
  bool result2D = runAll<2>({256, 256}, {32, 32}, 20, seed);

  if (result1D && result2D) {
    std::cout << "All tests passed!\n";
  } else {
    std::cerr << "An error has occurred!\n";
    return 1;
  }

  return 0;
}
