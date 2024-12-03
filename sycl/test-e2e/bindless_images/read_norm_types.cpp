// REQUIRES: aspect-ext_oneapi_bindless_images

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

#include <iostream>
#include <limits>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include "helpers/common.hpp"
#include <sycl/ext/oneapi/bindless_images.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

static sycl::device dev;

template <int NDims, typename DType, int NChannels,
          sycl::image_channel_type CType, typename KernelName>
bool run_test(sycl::range<NDims> globalSize, sycl::range<NDims> localSize) {
  using InputType = sycl::vec<DType, NChannels>;
  using OutputType = sycl::vec<float, NChannels>;

  sycl::queue q(dev);
  auto ctxt = q.get_context();

  size_t numElems = globalSize.size();
  constexpr auto dtypeMaxVal = std::numeric_limits<DType>::max();

  std::vector<InputType> dataIn(numElems, InputType((DType)dtypeMaxVal));
  std::vector<OutputType> dataOut(numElems);
  std::vector<OutputType> expected(numElems, OutputType(2.f));

  try {

    namespace syclexp = sycl::ext::oneapi::experimental;

    syclexp::image_descriptor descIn(globalSize, NChannels, CType);
    syclexp::image_descriptor descOut(globalSize, NChannels,
                                      sycl::image_channel_type::fp32);

    syclexp::image_mem_handle imgMemIn = syclexp::alloc_image_mem(descIn, q);
    syclexp::image_mem_handle imgMemOut = syclexp::alloc_image_mem(descOut, q);

    syclexp::bindless_image_sampler sampler{
        sycl::addressing_mode::clamp,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::nearest};

    auto imgIn1 = syclexp::create_image(imgMemIn, sampler, descIn, q);
    auto imgOut = syclexp::create_image(imgMemOut, descOut, q);

    void *allocUSM = nullptr;
    syclexp::image_mem_handle allocMem;
    syclexp::sampled_image_handle imgIn2;

    if constexpr (NDims == 2) {
      size_t pitch = 0;
      allocUSM = syclexp::pitched_alloc_device(&pitch, descIn, q);

      if (allocUSM == nullptr) {
        std::cerr << "Error allocating 2D USM memory!" << std::endl;
        return false;
      }
      imgIn2 = syclexp::create_image(allocUSM, pitch, sampler, descIn, q);
      q.ext_oneapi_copy(dataIn.data(), allocUSM, descIn, pitch);

    } else {
      allocMem = syclexp::alloc_image_mem(descIn, q);
      imgIn2 = syclexp::create_image(allocMem, sampler, descIn, q);
      q.ext_oneapi_copy(dataIn.data(), allocMem, descIn);
    }

    q.ext_oneapi_copy(dataIn.data(), imgMemIn, descIn);
    q.wait_and_throw();

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<KernelName>(
          sycl::nd_range<NDims>{globalSize, localSize},
          [=](sycl::nd_item<NDims> it) {
            if constexpr (NDims == 1) {
              size_t dim0 = it.get_global_id(0);
              float fdim0 = dim0 / globalSize[0];
              OutputType pixel1 =
                  syclexp::sample_image<OutputType>(imgIn1, fdim0);
              OutputType pixel2 =
                  syclexp::sample_image<OutputType>(imgIn2, fdim0);
              syclexp::write_image(imgOut, int(dim0), pixel1 + pixel2);
            } else if constexpr (NDims == 2) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);
              float fdim0 = dim0 / globalSize[0];
              float fdim1 = dim1 / globalSize[1];
              OutputType pixel1 = syclexp::sample_image<OutputType>(
                  imgIn1, sycl::float2(fdim0, fdim1));
              OutputType pixel2 = syclexp::sample_image<OutputType>(
                  imgIn2, sycl::float2(fdim0, fdim1));
              syclexp::write_image(imgOut, sycl::int2(dim0, dim1),
                                   pixel1 + pixel2);
            } else if constexpr (NDims == 3) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);
              size_t dim2 = it.get_global_id(2);
              float fdim0 = dim0 / globalSize[0];
              float fdim1 = dim1 / globalSize[1];
              float fdim2 = dim2 / globalSize[2];
              OutputType pixel1 = syclexp::sample_image<OutputType>(
                  imgIn1, sycl::float3(fdim0, fdim1, fdim2));
              OutputType pixel2 = syclexp::sample_image<OutputType>(
                  imgIn2, sycl::float3(fdim0, fdim1, fdim2));
              syclexp::write_image(imgOut, sycl::int3(dim0, dim1, dim2),
                                   pixel1 + pixel2);
            }
          });
    });

    q.wait_and_throw();
    q.ext_oneapi_copy(imgMemOut, dataOut.data(), descOut);
    q.wait_and_throw();

    syclexp::destroy_image_handle(imgIn1, q);
    syclexp::destroy_image_handle(imgIn2, q);
    syclexp::destroy_image_handle(imgOut, q);

    syclexp::free_image_mem(imgMemIn, syclexp::image_type::standard, dev, ctxt);
    syclexp::free_image_mem(imgMemOut, syclexp::image_type::standard, dev,
                            ctxt);

    if constexpr (NDims == 2) {
      sycl::free(allocUSM, ctxt);
    } else {
      syclexp::free_image_mem(allocMem, syclexp::image_type::standard, dev,
                              ctxt);
    }
  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return false;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return false;
  }

  // collect and validate output
  bool validated = true;
  for (int i = 0; i < globalSize.size(); i++) {
    bool mismatch = false;
    if (!bindless_helpers::equal_vec<float, NChannels>(dataOut[i],
                                                       expected[i])) {
      mismatch = true;
      validated = false;
    }
    if (mismatch) {
#ifdef VERBOSE_PRINT
      std::cout << "Result mismatch! Expected: " << expected[i][0]
                << ", Actual: " << dataOut[i][0] << std::endl;
#else
      break;
#endif
    }
  }
  if (validated) {
    return true;
  }

  return false;
}

void verbose_println(std::string message) {
#ifdef VERBOSE_PRINT
  std::cout << message << "\n";
#endif
}

int main() {

  bool validated = true;

  // 1D tests
  verbose_println("Running kernel unorm_int8_1d_1c");
  validated &= run_test<1, uint8_t, 1, sycl::image_channel_type::unorm_int8,
                        class unorm_int8_1d_1c>({32}, {2});
  verbose_println("Running kernel snorm_int8_1d_1c");
  validated &= run_test<1, int8_t, 1, sycl::image_channel_type::snorm_int8,
                        class snorm_int8_1d_1c>({32}, {2});
  verbose_println("Running kernel unorm_int16_1d_1c");
  validated &= run_test<1, uint16_t, 1, sycl::image_channel_type::unorm_int16,
                        class unorm_int16_1d_1c>({32}, {2});
  verbose_println("Running kernel snorm_int16_1d_1c");
  validated &= run_test<1, int16_t, 1, sycl::image_channel_type::snorm_int16,
                        class snorm_int16_1d_1c>({32}, {2});

  verbose_println("Running kernel unorm_int8_1d_2c");
  validated &= run_test<1, uint8_t, 2, sycl::image_channel_type::unorm_int8,
                        class unorm_int8_1d_2c>({32}, {2});
  verbose_println("Running kernel snorm_int8_1d_2c");
  validated &= run_test<1, int8_t, 2, sycl::image_channel_type::snorm_int8,
                        class snorm_int8_1d_2c>({32}, {2});
  verbose_println("Running kernel unorm_int16_1d_2c");
  validated &= run_test<1, uint16_t, 2, sycl::image_channel_type::unorm_int16,
                        class unorm_int16_1d_2c>({32}, {2});
  verbose_println("Running kernel snorm_int16_1d_2c");
  validated &= run_test<1, int16_t, 2, sycl::image_channel_type::snorm_int16,
                        class snorm_int16_1d_2c>({32}, {2});

  verbose_println("Running kernel unorm_int8_1d_4c");
  validated &= run_test<1, uint8_t, 4, sycl::image_channel_type::unorm_int8,
                        class unorm_int8_1d_4c>({32}, {2});
  verbose_println("Running kernel snorm_int8_1d_4c");
  validated &= run_test<1, int8_t, 4, sycl::image_channel_type::snorm_int8,
                        class snorm_int8_1d_4c>({32}, {2});
  verbose_println("Running kernel unorm_int16_1d_4c");
  validated &= run_test<1, uint16_t, 4, sycl::image_channel_type::unorm_int16,
                        class unorm_int16_1d_4c>({32}, {2});
  verbose_println("Running kernel snorm_int16_1d_4c");
  validated &= run_test<1, int16_t, 4, sycl::image_channel_type::snorm_int16,
                        class snorm_int16_1d_4c>({32}, {2});

  // 2D tests
  verbose_println("Running kernel unorm_int8_2d_1c");
  validated &= run_test<2, uint8_t, 1, sycl::image_channel_type::unorm_int8,
                        class unorm_int8_2d_1c>({32, 32}, {16, 16});
  verbose_println("Running kernel snorm_int8_2d_1c");
  validated &= run_test<2, int8_t, 1, sycl::image_channel_type::snorm_int8,
                        class snorm_int8_2d_1c>({32, 32}, {16, 16});
  verbose_println("Running kernel unorm_int16_2d_1c");
  validated &= run_test<2, uint16_t, 1, sycl::image_channel_type::unorm_int16,
                        class unorm_int16_2d_1c>({32, 32}, {16, 16});
  verbose_println("Running kernel snorm_int16_2d_1c");
  validated &= run_test<2, int16_t, 1, sycl::image_channel_type::snorm_int16,
                        class snorm_int16_2d_1c>({32, 32}, {16, 16});

  verbose_println("Running kernel unorm_int8_2d_2c");
  validated &= run_test<2, uint8_t, 2, sycl::image_channel_type::unorm_int8,
                        class unorm_int8_2d_2c>({32, 32}, {16, 16});
  verbose_println("Running kernel snorm_int8_2d_2c");
  validated &= run_test<2, int8_t, 2, sycl::image_channel_type::snorm_int8,
                        class snorm_int8_2d_2c>({32, 32}, {16, 16});
  verbose_println("Running kernel unorm_int16_2d_2c");
  validated &= run_test<2, uint16_t, 2, sycl::image_channel_type::unorm_int16,
                        class unorm_int16_2d_2c>({32, 32}, {16, 16});
  verbose_println("Running kernel snorm_int16_2d_2c");
  validated &= run_test<2, int16_t, 2, sycl::image_channel_type::snorm_int16,
                        class snorm_int16_2d_2c>({32, 32}, {16, 16});

  verbose_println("Running kernel unorm_int8_2d_4c");
  validated &= run_test<2, uint8_t, 4, sycl::image_channel_type::unorm_int8,
                        class unorm_int8_2d_4c>({32, 32}, {16, 16});
  verbose_println("Running kernel snorm_int8_2d_4c");
  validated &= run_test<2, int8_t, 4, sycl::image_channel_type::snorm_int8,
                        class snorm_int8_2d_4c>({32, 32}, {16, 16});
  verbose_println("Running kernel unorm_int16_2d_4c");
  validated &= run_test<2, uint16_t, 4, sycl::image_channel_type::unorm_int16,
                        class unorm_int16_2d_4c>({32, 32}, {16, 16});
  verbose_println("Running kernel snorm_int16_2d_4c");
  validated &= run_test<2, int16_t, 4, sycl::image_channel_type::snorm_int16,
                        class snorm_int16_2d_4c>({32, 32}, {16, 16});

  // 3D tests
  verbose_println("Running kernel unorm_int8_3d_1c");
  validated &= run_test<3, uint8_t, 1, sycl::image_channel_type::unorm_int8,
                        class unorm_int8_3d_1c>({32, 32, 32}, {16, 16, 4});
  verbose_println("Running kernel snorm_int8_3d_1c");
  validated &= run_test<3, int8_t, 1, sycl::image_channel_type::snorm_int8,
                        class snorm_int8_3d_1c>({32, 32, 32}, {16, 16, 4});
  verbose_println("Running kernel unorm_int16_3d_1c");
  validated &= run_test<3, uint16_t, 1, sycl::image_channel_type::unorm_int16,
                        class unorm_int16_3d_1c>({32, 32, 32}, {16, 16, 4});
  verbose_println("Running kernel snorm_int16_3d_1c");
  validated &= run_test<3, int16_t, 1, sycl::image_channel_type::snorm_int16,
                        class snorm_int16_3d_1c>({32, 32, 32}, {16, 16, 4});

  verbose_println("Running kernel unorm_int8_3d_2c");
  validated &= run_test<3, uint8_t, 2, sycl::image_channel_type::unorm_int8,
                        class unorm_int8_3d_2c>({32, 32, 32}, {16, 16, 4});
  verbose_println("Running kernel snorm_int8_3d_2c");
  validated &= run_test<3, int8_t, 2, sycl::image_channel_type::snorm_int8,
                        class snorm_int8_3d_2c>({32, 32, 32}, {16, 16, 4});
  verbose_println("Running kernel unorm_int16_3d_2c");
  validated &= run_test<3, uint16_t, 2, sycl::image_channel_type::unorm_int16,
                        class unorm_int16_3d_2c>({32, 32, 32}, {16, 16, 4});
  verbose_println("Running kernel snorm_int16_3d_2c");
  validated &= run_test<3, int16_t, 2, sycl::image_channel_type::snorm_int16,
                        class snorm_int16_3d_2c>({32, 32, 32}, {16, 16, 4});

  verbose_println("Running kernel unorm_int8_3d_4c");
  validated &= run_test<3, uint8_t, 4, sycl::image_channel_type::unorm_int8,
                        class unorm_int8_3d_4c>({32, 32, 32}, {16, 16, 4});
  verbose_println("Running kernel snorm_int8_3d_4c");
  validated &= run_test<3, int8_t, 4, sycl::image_channel_type::snorm_int8,
                        class snorm_int8_3d_4c>({32, 32, 32}, {16, 16, 4});
  verbose_println("Running kernel unorm_int16_3d_4c");
  validated &= run_test<3, uint16_t, 4, sycl::image_channel_type::unorm_int16,
                        class unorm_int16_3d_4c>({32, 32, 32}, {16, 16, 4});
  verbose_println("Running kernel snorm_int16_3d_4c");
  validated &= run_test<3, int16_t, 4, sycl::image_channel_type::snorm_int16,
                        class snorm_int16_3d_4c>({32, 32, 32}, {16, 16, 4});

  if (validated)
    std::cout << "All test cases passed!\n";
  else
    std::cerr << "Some test cases failed\n";

  return !validated;
}
