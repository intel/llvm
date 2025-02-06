// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <numeric>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/usm.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

namespace syclexp = sycl::ext::oneapi::experimental;

template <typename typeSrc, typename typeDest>
void copy_image_mem_handle_to_image_mem_handle(
    syclexp::image_descriptor &descSrc, syclexp::image_descriptor &descDest,
    const std::vector<typeSrc> &testData, sycl::device dev, sycl::queue q,
    std::vector<typeDest> &out) {
  syclexp::image_mem_handle imgMemSrc = syclexp::alloc_image_mem(descSrc, q);
  syclexp::image_mem_handle imgMemDst = syclexp::alloc_image_mem(descDest, q);

  q.ext_oneapi_copy(testData.data(), imgMemSrc, descSrc);
  q.wait_and_throw();

  q.ext_oneapi_copy(imgMemSrc, descSrc, imgMemDst, descDest);
  q.wait_and_throw();

  q.ext_oneapi_copy(imgMemDst, out.data(), descDest);
  q.wait_and_throw();

  syclexp::free_image_mem(imgMemSrc, syclexp::image_type::standard, q);
  syclexp::free_image_mem(imgMemDst, syclexp::image_type::standard, q);
}

template <int NDims, typename typeSrc, typename typeDest>
void copy_image_mem_handle_to_usm(syclexp::image_descriptor &descSrc,
                                  syclexp::image_descriptor &descDest,
                                  const std::vector<typeSrc> &testData,
                                  sycl::device dev, sycl::queue q,
                                  std::vector<typeDest> &out) {
  syclexp::image_mem imgMemSrc(descSrc, dev, q.get_context());

  size_t pitch = 0;
  void *imgMemDst = nullptr;
  if (NDims == 1) {
    size_t elements = descDest.width * descDest.num_channels;
    imgMemDst = sycl::malloc_device<typeDest>(elements, q);
  } else {
    imgMemDst = syclexp::pitched_alloc_device(&pitch, descDest, q);
  }

  q.ext_oneapi_copy(testData.data(), imgMemSrc.get_handle(), descSrc);
  q.wait_and_throw();

  q.ext_oneapi_copy(imgMemSrc.get_handle(), descSrc, imgMemDst, descDest,
                    pitch);
  q.wait_and_throw();

  q.ext_oneapi_copy(imgMemDst, out.data(), descDest, pitch);
  q.wait_and_throw();

  sycl::free(imgMemDst, q.get_context());
}

template <int NDims, typename typeSrc, typename typeDest>
void copy_usm_to_image_mem_handle(syclexp::image_descriptor &descSrc,
                                  syclexp::image_descriptor &descDest,
                                  const std::vector<typeSrc> &testData,
                                  sycl::device dev, sycl::queue q,
                                  std::vector<typeDest> &out) {
  size_t pitch = 0;
  void *imgMemSrc = nullptr;
  if (NDims == 1) {
    size_t elements = descSrc.width * descSrc.num_channels;
    imgMemSrc = sycl::malloc_device<typeSrc>(elements, q);
  } else {
    imgMemSrc = syclexp::pitched_alloc_device(&pitch, descSrc, q);
  }

  syclexp::image_mem imgMemDst(descDest, dev, q.get_context());

  q.ext_oneapi_copy(testData.data(), imgMemSrc, descSrc, pitch);
  q.wait_and_throw();

  q.ext_oneapi_copy(imgMemSrc, descSrc, pitch, imgMemDst.get_handle(),
                    descDest);
  q.wait_and_throw();

  q.ext_oneapi_copy(imgMemDst.get_handle(), out.data(), descDest);
  q.wait_and_throw();

  sycl::free(imgMemSrc, q.get_context());
}

template <int NDims, typename typeSrc, typename typeDest>
void copy_usm_to_usm(syclexp::image_descriptor &descSrc,
                     syclexp::image_descriptor &descDest,
                     const std::vector<typeSrc> &testData, sycl::device dev,
                     sycl::queue q, std::vector<typeDest> &out) {

  size_t pitchSrc = 0;
  void *imgMemSrc = nullptr;
  if (NDims == 1) {
    size_t elements = descSrc.width * descSrc.num_channels;
    imgMemSrc = sycl::malloc_device<typeSrc>(elements, q);
  } else {
    imgMemSrc = syclexp::pitched_alloc_device(&pitchSrc, descSrc, q);
  }

  size_t pitchDest = 0;
  void *imgMemDst = nullptr;
  if (NDims == 1) {
    size_t elements = descDest.width * descDest.num_channels;
    imgMemDst = sycl::malloc_device<typeDest>(elements, q);
  } else {
    imgMemDst = syclexp::pitched_alloc_device(&pitchDest, descDest, q);
  }

  q.ext_oneapi_copy(testData.data(), imgMemSrc, descSrc, pitchSrc);
  q.wait_and_throw();

  q.ext_oneapi_copy(imgMemSrc, descSrc, pitchSrc, imgMemDst, descDest,
                    pitchDest);
  q.wait_and_throw();

  q.ext_oneapi_copy(imgMemDst, out.data(), descDest, pitchDest);
  q.wait_and_throw();

  sycl::free(imgMemSrc, q.get_context());
  sycl::free(imgMemDst, q.get_context());
}

template <typename type>
bool check_test(const std::vector<type> &out,
                const std::vector<type> &expected) {
  assert(out.size() == expected.size());
  bool validated = true;
  for (int i = 0; i < out.size(); i++) {
    bool mismatch = false;
    if (out[i] != expected[i]) {
      mismatch = true;
      validated = false;
    }

    if (mismatch) {
#ifdef VERBOSE_PRINT
      std::cout << "Result mismatch! Expected: " << expected[i]
                << ", Actual: " << out[i] << std::endl;
#else
      break;
#endif
    }
  }
  return validated;
}

template <int channelNumSrc, sycl::image_channel_type channelTypeSrc,
          typename dataTypeSrc, int channelNumDest,
          sycl::image_channel_type channelTypeDest, typename dataTypeDest,
          int NDims,
          syclexp::image_type imageType = syclexp::image_type::standard>
bool run_copy_test_with(sycl::device &dev, sycl::queue &q,
                        sycl::range<NDims> dims) {
  static_assert(sizeof(dataTypeSrc) == sizeof(dataTypeDest),
                "Image data type sizes aren't equal");
  std::vector<dataTypeSrc> dataSequence(dims.size());
  std::vector<dataTypeDest> out(dims.size());
  std::vector<dataTypeDest> expected(dims.size());

  std::iota(dataSequence.begin(), dataSequence.end(), 0);

  if constexpr (std::is_same_v<dataTypeSrc, dataTypeDest>) {
    std::iota(expected.begin(), expected.end(), 0);
  } else {
    std::memcpy(expected.data(), dataSequence.data(),
                dims.size() * sizeof(dataTypeDest));
  }

  syclexp::image_descriptor descSrc;
  syclexp::image_descriptor descDest;

  if constexpr (imageType == syclexp::image_type::standard) {
    descSrc = syclexp::image_descriptor(dims, channelNumSrc, channelTypeSrc);
    descDest = syclexp::image_descriptor(dims, channelNumDest, channelTypeDest);
  } else {
    descSrc = syclexp::image_descriptor(
        {dims[0], NDims > 2 ? dims[1] : 0}, channelNumSrc, channelTypeSrc,
        syclexp::image_type::array, 1, NDims > 2 ? dims[2] : dims[1]);
    descDest = syclexp::image_descriptor(
        {dims[0], NDims > 2 ? dims[1] : 0}, channelNumDest, channelTypeDest,
        syclexp::image_type::array, 1, NDims > 2 ? dims[2] : dims[1]);
  }

  bool verified = true;

  copy_image_mem_handle_to_image_mem_handle<dataTypeSrc, dataTypeDest>(
      descSrc, descDest, dataSequence, dev, q, out);
  verified = verified && check_test<dataTypeDest>(out, expected);

  // 3D USM image memory and image arrays backed by USM are not supported
  if (NDims != 3 && imageType != syclexp::image_type::array) {
    copy_image_mem_handle_to_usm<NDims, dataTypeSrc, dataTypeDest>(
        descSrc, descDest, dataSequence, dev, q, out);
    verified = verified && check_test<dataTypeDest>(out, expected);

    copy_usm_to_image_mem_handle<NDims, dataTypeSrc, dataTypeDest>(
        descSrc, descDest, dataSequence, dev, q, out);
    verified = verified && check_test<dataTypeDest>(out, expected);

    copy_usm_to_usm<NDims, dataTypeSrc, dataTypeDest>(
        descSrc, descDest, dataSequence, dev, q, out);
    verified = verified && check_test<dataTypeDest>(out, expected);
  }

  return verified;
}

template <int channelNum, sycl::image_channel_type channelType,
          typename imageType, int NDims,
          syclexp::image_type type = syclexp::image_type::standard>
bool run_copy_test_with(sycl::device &dev, sycl::queue &q,
                        sycl::range<NDims> dims) {
  return run_copy_test_with<channelNum, channelType, imageType, channelNum,
                            channelType, imageType, NDims, type>(dev, q, dims);
}

int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // Standard images copies
  bool validated =
      run_copy_test_with<1, sycl::image_channel_type::fp32, float, 2>(
          dev, q, {128, 128});

  validated &= run_copy_test_with<1, sycl::image_channel_type::fp32, float, 1>(
      dev, q, {128 * 4});

  validated &= run_copy_test_with<1, sycl::image_channel_type::fp32, float, 3>(
      dev, q, {128, 128, 4});

  // Standard image copies using different data types
  validated &=
      run_copy_test_with<1, sycl::image_channel_type::fp32, float, 1,
                         sycl::image_channel_type::signed_int32, int, 2>(
          dev, q, {128, 128});

  if (dev.has(sycl::aspect::fp16)) {
    validated &=
        run_copy_test_with<1, sycl::image_channel_type::fp16, sycl::half, 1,
                           sycl::image_channel_type::signed_int16, short, 2>(
            dev, q, {128, 128});
  }

  // Layered images copies
  validated &=
      run_copy_test_with<1, sycl::image_channel_type::fp32, float, 2,
                         syclexp::image_type::array>(dev, q, {956, 38});

  validated &=
      run_copy_test_with<1, sycl::image_channel_type::fp32, float, 3,
                         syclexp::image_type::array>(dev, q, {128, 128, 4});

  // Layered image copies using different data types
  validated &=
      run_copy_test_with<1, sycl::image_channel_type::fp32, float, 1,
                         sycl::image_channel_type::signed_int32, int, 2,
                         syclexp::image_type::array>(dev, q, {956, 38});

  validated &=
      run_copy_test_with<1, sycl::image_channel_type::unsigned_int32,
                         unsigned int, 1, sycl::image_channel_type::fp32, float,
                         2, syclexp::image_type::array>(dev, q, {956, 38});

  if (!validated) {
    std::cout << "Tests failed\n";
    return 1;
  }

  std::cout << "Tests passed\n";

  return 0;
}
