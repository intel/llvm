// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %t.out

// Print test names and pass status
// #define VERBOSE_LV1

// Same as above plus sampler, offset, margin of error, largest error found and
// results of one mismatch
// #define VERBOSE_LV2

// Same as above but all mismatches are printed
// #define VERBOSE_LV3

#include "helpers/common.hpp"
#include "helpers/sampling.hpp"
#include <cassert>
#include <iostream>
#include <random>
#include <sycl/accessor_image.hpp>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

namespace util {
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

      VecType result = sampling_helpers::read<NDims, DType, NChannels>(
          globalSizeTwoComp, coords, offset, samp, inputImage);
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
            sycl::vec<float, NDims> coords;

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
          sycl::image_channel_type CType, typename KernelName>
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
    // Check default constructor for image_descriptor
    syclexp::image_descriptor desc;
    desc = syclexp::image_descriptor(dims, NChannels, CType);

    syclexp::image_mem imgMem(desc, q);

    // Check that image_mem_handle can be constructed from raw_handle_type
    syclexp::image_mem_handle img_mem_handle_copy{
        static_cast<syclexp::image_mem_handle::raw_handle_type>(
            imgMem.get_handle().raw_handle)};
    if (img_mem_handle_copy.raw_handle != imgMem.get_handle().raw_handle) {
      std::cerr << "Failed to copy raw_handle_type" << std::endl;
      return false;
    }

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
        if (!sampling_helpers::isNumberWithinPercentOfNumber(
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

      bindless_helpers::printTestName<NDims>("Running 1D short", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, short, 1, sycl::image_channel_type::signed_int16,
                        class short_1d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 1D short2", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, short, 2, sycl::image_channel_type::signed_int16,
                        class short2_1d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 1D short4", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, short, 4, sycl::image_channel_type::signed_int16,
                        class short4_1d>(dims, localSize, offset, samp, seed);

      bindless_helpers::printTestName<NDims>("Running 1D unsigned short", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, unsigned short, 1,
                        sycl::image_channel_type::unsigned_int16,
                        class ushort_1d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 1D unsigned short2", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, unsigned short, 2,
                        sycl::image_channel_type::unsigned_int16,
                        class ushort2_1d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 1D unsigned short4", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, unsigned short, 4,
                        sycl::image_channel_type::unsigned_int16,
                        class ushort4_1d>(dims, localSize, offset, samp, seed);

      bindless_helpers::printTestName<NDims>("Running 1D char", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, signed char, 1,
                        sycl::image_channel_type::signed_int8, class char_1d>(
              dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 1D char2", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, signed char, 2,
                        sycl::image_channel_type::signed_int8, class char2_1d>(
              dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 1D char4", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, signed char, 4,
                        sycl::image_channel_type::signed_int8, class char4_1d>(
              dims, localSize, offset, samp, seed);

      bindless_helpers::printTestName<NDims>("Running 1D unsigned char", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, unsigned char, 1,
                        sycl::image_channel_type::unsigned_int8,
                        class uchar_1d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 1D unsigned char2", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, unsigned char, 2,
                        sycl::image_channel_type::unsigned_int8,
                        class uchar2_1d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 1D unsigned char4", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, unsigned char, 4,
                        sycl::image_channel_type::unsigned_int8,
                        class uchar4_1d>(dims, localSize, offset, samp, seed);

      bindless_helpers::printTestName<NDims>("Running 1D float", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, float, 1, sycl::image_channel_type::fp32,
                        class float_1d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 1D float2", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, float, 2, sycl::image_channel_type::fp32,
                        class float2_1d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 1D float4", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, float, 4, sycl::image_channel_type::fp32,
                        class float4_1d>(dims, localSize, offset, samp, seed);

      bindless_helpers::printTestName<NDims>("Running 1D half", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, sycl::half, 1, sycl::image_channel_type::fp16,
                        class half_1d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 1D half2", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, sycl::half, 2, sycl::image_channel_type::fp16,
                        class half2_1d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 1D half4", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, sycl::half, 4, sycl::image_channel_type::fp16,
                        class half4_1d>(dims, localSize, offset, samp, seed);

      bindless_helpers::printTestName<NDims>("Running 1D float", {512}, {32});
      failed |= util::runTest<NDims, float, 1, sycl::image_channel_type::fp32,
                              class float_1d1>({512}, {32}, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 1D float4", {512}, {8});
      failed |= util::runTest<NDims, float, 4, sycl::image_channel_type::fp32,
                              class float4_1d2>({512}, {8}, offset, samp, seed);
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

      bindless_helpers::printTestName<NDims>("Running 2D short", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, short, 1, sycl::image_channel_type::signed_int16,
                        class short_2d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 2D short2", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, short, 2, sycl::image_channel_type::signed_int16,
                        class short2_2d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 2D short4", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, short, 4, sycl::image_channel_type::signed_int16,
                        class short4_2d>(dims, localSize, offset, samp, seed);

      bindless_helpers::printTestName<NDims>("Running 2D unsigned short", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, unsigned short, 1,
                        sycl::image_channel_type::unsigned_int16,
                        class ushort_2d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 2D unsigned short2", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, unsigned short, 2,
                        sycl::image_channel_type::unsigned_int16,
                        class ushort2_2d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 2D unsigned short4", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, unsigned short, 4,
                        sycl::image_channel_type::unsigned_int16,
                        class ushort4_2d>(dims, localSize, offset, samp, seed);

      bindless_helpers::printTestName<NDims>("Running 2D char", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, signed char, 1,
                        sycl::image_channel_type::signed_int8, class char_2d>(
              dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 2D char2", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, signed char, 2,
                        sycl::image_channel_type::signed_int8, class char2_2d>(
              dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 2D char4", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, signed char, 4,
                        sycl::image_channel_type::signed_int8, class char4_2d>(
              dims, localSize, offset, samp, seed);

      bindless_helpers::printTestName<NDims>("Running 2D unsigned char", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, unsigned char, 1,
                        sycl::image_channel_type::unsigned_int8,
                        class uchar_2d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 2D unsigned char2", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, unsigned char, 2,
                        sycl::image_channel_type::unsigned_int8,
                        class uchar2_2d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 2D unsigned char4", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, unsigned char, 4,
                        sycl::image_channel_type::unsigned_int8,
                        class uchar4_2d>(dims, localSize, offset, samp, seed);

      bindless_helpers::printTestName<NDims>("Running 2D float", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, float, 1, sycl::image_channel_type::fp32,
                        class float_2d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 2D float2", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, float, 2, sycl::image_channel_type::fp32,
                        class float2_2d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 2D float4", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, float, 4, sycl::image_channel_type::fp32,
                        class float4_2d>(dims, localSize, offset, samp, seed);

      bindless_helpers::printTestName<NDims>("Running 2D half", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, sycl::half, 1, sycl::image_channel_type::fp16,
                        class half_2d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 2D half2", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, sycl::half, 2, sycl::image_channel_type::fp16,
                        class half2_2d>(dims, localSize, offset, samp, seed);
      bindless_helpers::printTestName<NDims>("Running 2D half4", dims,
                                             localSize);
      failed |=
          util::runTest<NDims, sycl::half, 4, sycl::image_channel_type::fp16,
                        class half4_2d>(dims, localSize, offset, samp, seed);

      bindless_helpers::printTestName<NDims>("Running 2D float", {512, 512},
                                             {32, 32});
      failed |= util::runTest<NDims, float, 1, sycl::image_channel_type::fp32,
                              class float_2d1>({512, 512}, {32, 32}, offset,
                                               samp, seed);
      bindless_helpers::printTestName<NDims>("Running 2D float4", {512, 512},
                                             {8, 8});
      failed |= util::runTest<NDims, float, 4, sycl::image_channel_type::fp32,
                              class float4_2d2>({512, 512}, {8, 8}, offset,
                                                samp, seed);
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
