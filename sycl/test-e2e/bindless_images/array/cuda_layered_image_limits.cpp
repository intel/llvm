// REQUIRES: gpu, target-nvidia, cuda_dev_kit
// REQUIRES: aspect-ext_oneapi_bindless_images, aspect-ext_oneapi_image_array
// RUN: %{build} %cuda_options -o %t.out
// RUN: %{run} %t.out

#define SYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL 1
#include <cuda.h>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

#include <algorithm>
#include <iostream>

namespace exp = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue q;
  CUdevice device =
      sycl::get_native<sycl::backend::ext_oneapi_cuda>(q.get_device());
  int textureLayers = 0, surfaceLayers = 0;
  if (cuDeviceGetAttribute(&textureLayers,
                           CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
                           device) != CUDA_SUCCESS ||
      cuDeviceGetAttribute(&surfaceLayers,
                           CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS,
                           device) != CUDA_SUCCESS)
    return 1;
  size_t maxLayers = std::min(textureLayers, surfaceLayers);
  if (maxLayers < 2)
    return 1;

  for (size_t layers : {size_t{2}, maxLayers, maxLayers + 1}) {
    exp::image_descriptor desc({1, 1}, 1, sycl::image_channel_type::fp32,
                               exp::image_type::array, 1, layers);
    auto memoryTypes = exp::get_image_memory_support(desc, q);
    auto opaque = exp::image_memory_handle_type::opaque_handle;
    bool memorySupported =
        std::find(memoryTypes.begin(), memoryTypes.end(), opaque) !=
        memoryTypes.end();
    bool sampledSupported =
        exp::is_image_handle_supported<exp::sampled_image_handle>(desc, opaque,
                                                                    q);
    bool unsampledSupported =
        exp::is_image_handle_supported<exp::unsampled_image_handle>(desc, opaque,
                                                                      q);
    bool expected = layers <= maxLayers;
    if (memorySupported != expected || sampledSupported != expected ||
        unsampledSupported != expected) {
      std::cerr << "Unexpected support for " << layers << " layers (limit "
                << maxLayers << ")\n";
      return 1;
    }
  }
  return 0;
}
