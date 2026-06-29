#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>

#include <detail/event_impl.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>
#include <sycl/queue.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

thread_local size_t ExpectedRowPitch = 0;
thread_local size_t ExpectedSlicePitch = 0;
thread_local uint32_t ExpectedNumSamples = 0;
thread_local int MapExternalArrayCallCounter = 0;

// -----
// Mocks
// -----
inline ur_result_t urBindlessImagesMapExternalArrayExp_replace(void *pParams) {
  ++MapExternalArrayCallCounter;
  auto Params =
      *reinterpret_cast<ur_bindless_images_map_external_array_exp_params_t *>(
          pParams);

  const ur_image_desc_t *urDesc = *Params.ppImageDesc;

  EXPECT_EQ(urDesc->rowPitch, ExpectedRowPitch);
  EXPECT_EQ(urDesc->slicePitch, ExpectedSlicePitch);
  EXPECT_EQ(urDesc->numSamples, ExpectedNumSamples);

  if (Params.pphImageMem && *Params.pphImageMem) {
    **Params.pphImageMem =
        mock::createDummyHandle<ur_exp_image_mem_native_handle_t>();
  }
  return UR_RESULT_SUCCESS;
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------
TEST(BindlessImagesExtensionTests, ImageDescriptorPropagatesLayout) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback(
      "urBindlessImagesMapExternalArrayExp",
      &urBindlessImagesMapExternalArrayExp_replace);

  sycl::queue Q;

  MapExternalArrayCallCounter = 0;
  ExpectedRowPitch = 1024;
  ExpectedSlicePitch = 0;
  ExpectedNumSamples = 4;

  syclexp::image_descriptor Desc(sycl::range<2>{32, 32}, 4,
                                 sycl::image_channel_type::fp32,
                                 syclexp::image_type::standard, 1, 1,
                                 ExpectedNumSamples, ExpectedRowPitch);

  syclexp::external_mem_descriptor<syclexp::resource_fd> ExtMemDesc{
      {123}, syclexp::external_mem_handle_type::opaque_fd, 0};

  auto MemHandle = syclexp::import_external_memory(ExtMemDesc, Q.get_device(),
                                                   Q.get_context());

  try {
    auto ImgHandle = syclexp::map_external_image_memory(MemHandle, Desc, Q);
    (void)ImgHandle;
  } catch (const sycl::exception &e) {
    FAIL() << "Caught unexpected SYCL exception: " << e.what();
  }

  EXPECT_EQ(MapExternalArrayCallCounter, 1);
}

TEST(BindlessImagesExtensionTests, ImageDescriptorPropagatesSlicePitch) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback(
      "urBindlessImagesMapExternalArrayExp",
      &urBindlessImagesMapExternalArrayExp_replace);

  sycl::queue Q;

  MapExternalArrayCallCounter = 0;
  ExpectedRowPitch = 512;
  ExpectedSlicePitch = 2048;
  ExpectedNumSamples = 1;

  syclexp::image_descriptor Desc(
      sycl::range<3>{16, 16, 4}, 1, sycl::image_channel_type::fp32,
      syclexp::image_type::standard, 1, 1, ExpectedNumSamples, ExpectedRowPitch,
      ExpectedSlicePitch);

  syclexp::external_mem_descriptor<syclexp::resource_fd> ExtMemDesc{
      {456}, syclexp::external_mem_handle_type::opaque_fd, 0};

  auto MemHandle = syclexp::import_external_memory(ExtMemDesc, Q.get_device(),
                                                   Q.get_context());

  try {
    auto ImgHandle = syclexp::map_external_image_memory(MemHandle, Desc, Q);
    (void)ImgHandle;
  } catch (const sycl::exception &e) {
    FAIL() << "Caught unexpected SYCL exception: " << e.what();
  }

  EXPECT_EQ(MapExternalArrayCallCounter, 1);
}