#pragma once
#include <random>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>

namespace sampling_helpers {
static bool isNumberWithinPercentOfNumber(float firstN, float percent,
                                          float secondN, float &diff,
                                          float &percDiff) {
  // Get absolute difference of the two numbers
  diff = std::abs(firstN - secondN);
  // Get the percentage difference of the two numbers
  percDiff = 100.0f * (diff / (((firstN + secondN) / 2.0f)));

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
  float pixelCoord{0.0f};

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

static void
printTestInfo(sycl::ext::oneapi::experimental::bindless_image_sampler &samp,
              float offset) {

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
                            const std::vector<VecType> &inputImage) {
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
                                  const std::vector<VecType> &inputImage) {
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
                             const std::vector<VecType> &inputImage) {
  int width = globalSize[0];

  int coordXInt =
      repeatNearestCoord(coords[0], static_cast<int>(globalSize[0]));
  int coordYInt =
      repeatNearestCoord(coords[1], static_cast<int>(globalSize[1]));

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
                                     const std::vector<VecType> &inputImage) {
  int width = globalSize[0];

  int coordXInt =
      mirroredRepeatNearestCoord(coords[0], static_cast<int>(globalSize[0]));
  int coordYInt =
      mirroredRepeatNearestCoord(coords[1], static_cast<int>(globalSize[1]));

  return inputImage[coordXInt + (width * coordYInt)];
}

template <typename VecType>
static VecType noneNearest(sycl::vec<float, 2> coords,
                           sycl::range<2> globalSize,
                           const std::vector<VecType> &inputImage) {
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
                                      const std::vector<VecType> &inputImage) {
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
            const std::vector<sycl::vec<DType, NChannels>> &inputImage) {
  using VecType = sycl::vec<DType, NChannels>;

  float coordX = coords[0];
  float coordY = coords[1];
  int width = globalSize[0];
  int height = globalSize[1];

  // Get coords for linear sampling
  int i0 = 0, i1 = 0;
  float weightX = getCommonLinearFractAndCoords(coordX, &i0, &i1);

  int j0 = 0, j1 = 0;
  // If height is not one, run as normal.
  // Otherwise, keep weightY set to 0.
  float weightY =
      height == 1 ? 0 : getCommonLinearFractAndCoords(coordY, &j0, &j1);

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
  return linearOp<NChannels, DType>(pix1, pix2, pix3, pix4, weightX, weightY);
}

// Out of range coords are clamped to the extent.
template <typename DType, int NChannels>
static sycl::vec<DType, NChannels>
clampToEdgeLinear(sycl::vec<float, 2> coords, sycl::range<2> globalSize,
                  const std::vector<sycl::vec<DType, NChannels>> &inputImage) {
  using VecType = sycl::vec<DType, NChannels>;

  float coordX = coords[0];
  float coordY = coords[1];
  int width = globalSize[0];
  int height = globalSize[1];

  // Get coords for linear sampling
  int i0, i1;
  float weightX = getCommonLinearFractAndCoords(coordX, &i0, &i1);

  int j0 = 0, j1 = 0;
  // If height is not one, run as normal.
  // Otherwise, keep weightY set to 0.
  float weightY =
      height == 1 ? 0 : getCommonLinearFractAndCoords(coordY, &j0, &j1);

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
  return linearOp<NChannels, DType>(pix1, pix2, pix3, pix4, weightX, weightY);
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
             const std::vector<sycl::vec<DType, NChannels>> &inputImage) {
  using VecType = sycl::vec<DType, NChannels>;

  float coordX = coords[0];
  float coordY = coords[1];
  int width = globalSize[0];
  int height = globalSize[1];

  InterpolRes resX = repeatLinearCoord(coordX, width);
  // If height is not one, run as normal.
  // Otherwise, set resY to all zeros.
  InterpolRes resY =
      height == 1 ? InterpolRes(0, 0, 0) : repeatLinearCoord(coordY, height);

  int i0 = resX.x0, i1 = resX.x1;
  int j0 = resY.x0, j1 = resY.x1;

  float weightX = resX.weight, weightY = resY.weight;

  // Wrap linear sampling coords to valid range
  i0 = repeatWrap(i0, width);
  i1 = repeatWrap(i1, width);
  j0 = repeatWrap(j0, height);
  j1 = repeatWrap(j1, height);

  VecType pix1 = inputImage[i0 + (width * j0)];
  VecType pix2 = inputImage[i1 + (width * j0)];
  VecType pix3 = inputImage[i0 + (width * j1)];
  VecType pix4 = inputImage[i1 + (width * j1)];

  // Perform linear sampling
  return linearOp<NChannels, DType>(pix1, pix2, pix3, pix4, weightX, weightY);
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
static sycl::vec<DType, NChannels> mirroredRepeatLinear(
    sycl::vec<float, 2> coords, sycl::range<2> globalSize,
    const std::vector<sycl::vec<DType, NChannels>> &inputImage) {
  using VecType = sycl::vec<DType, NChannels>;

  float coordX = coords[0];
  float coordY = coords[1];
  int width = globalSize[0];
  int height = globalSize[1];

  InterpolRes resX = mirroredRepeatLinearCoord(coordX, width);
  // If height is not one, run as normal.
  // Otherwise, set resY to all zeros.
  InterpolRes resY = height == 1 ? InterpolRes(0, 0, 0)
                                 : mirroredRepeatLinearCoord(coordY, height);

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
  return linearOp<NChannels, DType>(pix1, pix2, pix3, pix4, weightX, weightY);
}

// Some vector sizes here are hardcoded because the sampling functions are
// designed to only accept vecs of that size.
template <int NDims, typename DType, int NChannels>
static sycl::vec<DType, NChannels>
read(sycl::range<2> globalSize, sycl::vec<float, 2> coords, float offset,
     const sycl::ext::oneapi::experimental::bindless_image_sampler &samp,
     const std::vector<sycl::vec<DType, NChannels>> &inputImage) {
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
      return clampNearest<VecType>(coords, globalSize, inputImage);
    }

    if (SampAddrMode == sycl::addressing_mode::clamp_to_edge) {
      return clampToEdgeNearest<VecType>(coords, globalSize, inputImage);
    }

    if (SampAddrMode == sycl::addressing_mode::repeat) {
      if (SampNormMode == sycl::coordinate_normalization_mode::unnormalized) {
        assert(false &&
               "Repeat addressing mode must be used with normalized coords");
      }
      return repeatNearest<VecType>(coords, globalSize, inputImage);
    }

    if (SampAddrMode == sycl::addressing_mode::mirrored_repeat) {
      if (SampNormMode == sycl::coordinate_normalization_mode::unnormalized) {
        assert(false && "Mirrored repeat addressing mode must be used with "
                        "normalized coords");
      }
      return mirroredRepeatNearest<VecType>(coords, globalSize, inputImage);
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
      return noneNearest(coords, globalSize, inputImage);
    }

  } else { // linear
    sycl::addressing_mode SampAddrMode = samp.addressing[0];
    if (SampAddrMode == sycl::addressing_mode::clamp) {
      return clampLinear<DType, NChannels>(coords, globalSize, inputImage);
    }
    if (SampAddrMode == sycl::addressing_mode::clamp_to_edge) {
      return clampToEdgeLinear<DType, NChannels>(coords, globalSize,
                                                 inputImage);
    }
    if (SampAddrMode == sycl::addressing_mode::repeat) {
      if (SampNormMode == sycl::coordinate_normalization_mode::unnormalized) {
        assert(false &&
               "Repeat addressing mode must be used with normalized coords");
      }
      return repeatLinear<DType, NChannels>(coords, globalSize, inputImage);
    }
    if (SampAddrMode == sycl::addressing_mode::mirrored_repeat) {
      if (SampNormMode == sycl::coordinate_normalization_mode::unnormalized) {
        assert(false && "Mirrored repeat addressing mode must be used with "
                        "normalized coords");
      }
      return mirroredRepeatLinear<DType, NChannels>(coords, globalSize,
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

}; // namespace sampling_helpers
