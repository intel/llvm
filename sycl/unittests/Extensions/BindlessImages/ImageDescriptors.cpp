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

namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {

// We define this specialization to satisfy the linker.
template <>
external_mem import_external_memory<int>(external_mem_descriptor<int>,
                                         const sycl::device &,
                                         const sycl::context &) {

  // Mock: We cannot access the private constructor of external_mem.
  // But we know it contains a shared_ptr<detail::external_mem_impl>.
  // SO allocate a raw buffer of that size and cast it.
  // Since we mock the consumer (MapExternalArray), we just need a non-null
  // object that doesn't crash on copy/move.

  // Allocate raw storage for the object
  // sizeof(external_mem) is typically sizeof(shared_ptr) = 16 bytes (64-bit)
  // We allocate enough space to be safe.
  static char dummy_storage[64] = {0};

  // We pretend this storage is a valid external_mem object.
  return *reinterpret_cast<external_mem *>(dummy_storage);
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl

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

  int fd = 123;
  syclexp::external_mem_descriptor<int> ExtMemDesc{
      fd, syclexp::external_mem_handle_type::opaque_fd};

  // This calls our local dirty specialization
  auto MemHandle = syclexp::import_external_memory(ExtMemDesc, Q.get_device(),
                                                   Q.get_context());

  try {
    // Pass the dirty object. The Mock intercepts it before the runtime
    // tries to dereference the internal (null) pointer.
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

  int fd = 456;
  syclexp::external_mem_descriptor<int> ExtMemDesc{
      fd, syclexp::external_mem_handle_type::opaque_fd};

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