// REQUIRES: aspect-ext_oneapi_bindless_images

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

// This test demonstrates that bindless image coordinate conventions are absolute,
// not relative/symmetric. We test two interpretations of the same inputs data.
//
// Original Data Is Linear: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
//
// We interpret it as a 5x6 image:
// x ->  0   1   2   3   4
// y ↓   5   6   7   8   9
//      10  11  12  13  14
//      15  16  17  18  19
//      20  21  22  23  24
//      25  26  27  28  29
//
// Then we use two different image_descriptors (one with reversed range<>) and sample pixels with each,
// storing the samples in two linear output buffers. If the API is flexible/relative, then both 
// interpretations should work. But they don't. bindless images expect row-major coordinate semantics.
// This is important to know if you want performant coalesced memory access. 
//


#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

int main() {
  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  size_t width = 5;
  size_t height = 6;
  size_t N = width * height;

  // Single input dataset - iota sequence [0, 1, 2, 3, ..., 29]
  std::vector<sycl::float4> dataIn(N);
  for (size_t i = 0; i < N; i++) {
    dataIn[i] = sycl::float4(static_cast<float>(i), 0, 0, 0);
  }

  std::cout << "Testing coordinate order conventions with " << width << "x" << height
            << " image" << std::endl;
  std::cout << "Input data: [0, 1, 2, ..., " << (N-1) << "]" << std::endl;

  // Output buffers for both interpretations
  std::vector<float> output_correct(N);
  std::vector<float> output_reversed(N);

  try {
    syclex::bindless_image_sampler samp(
        sycl::addressing_mode::repeat,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::nearest);

    // =============================================================================
    // CORRECT CONVENTION: descriptor({width, height}), coords as (x, y)
    // =============================================================================
    syclex::image_descriptor desc_correct({width, height}, 4, sycl::image_channel_type::fp32);
    syclex::image_mem imgMem_correct(desc_correct, dev, ctxt);

    q.ext_oneapi_copy(dataIn.data(), imgMem_correct.get_handle(), desc_correct);
    q.wait_and_throw();

    syclex::sampled_image_handle imgHandle_correct = syclex::create_image(imgMem_correct, samp, desc_correct, dev, ctxt);

    // =============================================================================
    // REVERSED CONVENTION: descriptor({height, width}), coords as (y, x)
    // =============================================================================
    syclex::image_descriptor desc_reversed({height, width}, 4, sycl::image_channel_type::fp32);
    syclex::image_mem imgMem_reversed(desc_reversed, dev, ctxt);

    // Copy SAME data with reversed descriptor
    q.ext_oneapi_copy(dataIn.data(), imgMem_reversed.get_handle(), desc_reversed);
    q.wait_and_throw();

    syclex::sampled_image_handle imgHandle_reversed = syclex::create_image(imgMem_reversed, samp, desc_reversed, dev, ctxt);

    // =============================================================================
    // Sample from BOTH images in the SAME kernel
    // =============================================================================
    // SYCL 2020 spec (section 3.11.1-3.11.2) says LAST dimension of range is the fastest varying, so we use {height, width}
    sycl::range<2> output_range{height, width};
    sycl::buffer<float, 2> buf_correct((float *)output_correct.data(), output_range);
    sycl::buffer<float, 2> buf_reversed((float *)output_reversed.data(), output_range);

    q.submit([&](sycl::handler &cgh) {
      auto acc_correct = buf_correct.get_access<sycl::access_mode::write>(cgh, output_range);
      auto acc_reversed = buf_reversed.get_access<sycl::access_mode::write>(cgh, output_range);

      cgh.parallel_for<class proof_test>(
          output_range,
          [=](sycl::id<2> idx) {
            
            size_t y = idx[0];
            size_t x = idx[1];

            float x_norm = float(x + 0.5f) / float(width);
            float y_norm = float(y + 0.5f) / float(height);

            // CORRECT: Sample with (x, y) convention
            sycl::float4 px_correct = syclex::sample_image<sycl::float4>(
                imgHandle_correct, sycl::float2(x_norm, y_norm));  // coordinates as (x, y)
            acc_correct[sycl::id<2>{y, x}] = px_correct[0];

            // REVERSED: Sample with (y, x) convention            
            sycl::float4 px_reversed = syclex::sample_image<sycl::float4>(
                imgHandle_reversed, sycl::float2(y_norm, x_norm));  // coordinates as (y, x)
            acc_reversed[sycl::id<2>{y, x}] = px_reversed[0];
          });
    });

    q.wait_and_throw();

    syclex::destroy_image_handle(imgHandle_correct, dev, ctxt);
    syclex::destroy_image_handle(imgHandle_reversed, dev, ctxt);

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception: " << e.what() << "\n";
    return 1;
  }

  // =============================================================================
  // VERIFICATION
  // =============================================================================

  std::cout << "\n=== CORRECT CONVENTION TEST ===" << std::endl;
  std::cout << "descriptor({width=" << width << ", height=" << height
            << "}), coords as float2(x, y)" << std::endl;

  bool correct_is_iota = true;
  for (size_t i = 0; i < N; i++) {
    if (output_correct[i] != static_cast<float>(i)) {
      std::cout << "  Mismatch at index " << i << ": expected " << i
                << ", got " << output_correct[i] << std::endl;
      correct_is_iota = false;
      if (i > 10) break; // Don't spam too much
    }
  }

  if (correct_is_iota) {
    std::cout << "✓ PASSED: Produces iota sequence [0,1,2,...,29]" << std::endl;
  } else {
    std::cout << "✗ FAILED: Does not produce iota sequence!" << std::endl;
    std::cout << "First 10 values: ";
    for (int i = 0; i < 10; i++) {
      std::cout << output_correct[i] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "\n=== REVERSED CONVENTION TEST ===" << std::endl;
  std::cout << "descriptor({height=" << height << ", width=" << width
            << "}), coords as float2(y, x)" << std::endl;

  bool reversed_is_iota = true;
  for (size_t i = 0; i < N; i++) {
    if (output_reversed[i] != static_cast<float>(i)) {
      reversed_is_iota = false;
      break;
    }
  }

  if (!reversed_is_iota) {
    std::cout << "✓ PASSED: Does NOT produce iota sequence (as expected!)" << std::endl;
    std::cout << "First 10 values: ";
    for (int i = 0; i < 10; i++) {
      std::cout << output_reversed[i] << " ";
    }
    std::cout << "..." << std::endl;
  } else {
    std::cout << "✗ UNEXPECTED: Reversed convention also produces iota!" << std::endl;
    std::cout << "This would suggest a symmetric/relative API (unlikely)." << std::endl;
  }

  // =============================================================================
  // FINAL
  // =============================================================================

  if (correct_is_iota && !reversed_is_iota) {
    std::cout << "✓✓ TEST PASSED ✓✓" << std::endl;
    // image_descriptor dimensions are in (width, height) order
    // Coordinates are the same (x, y) order
    // Memory layout is x-major: memory[x + y*width]
    // API is not  symmetric/relative
    return 0;
  } else if (!correct_is_iota && reversed_is_iota) {
    std::cout << "FAIL: Understanding is reversed! Convention appears to be {height, width}" << std::endl;
    return 1;
  } else if (correct_is_iota && reversed_is_iota) {
    std::cout << "FAIL: API appears symmetric/relative, which is unexpected and not likely performant." << std::endl;
    return 1;
  } else {
    std::cout << "FAIL: Neither convention produced expected iota sequence. Something wrong." << std::endl;
    return 1;
  }
}
