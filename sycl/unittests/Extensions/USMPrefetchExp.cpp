#include <helpers/UrMock.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/usm.hpp>

#include <gtest/gtest.h>

static constexpr int N = 8;
static ur_usm_migration_flags_t urUSMPrefetchDirection = -1;

ur_result_t redefinedEnqueueUSMPrefetch(void *pParams) {
  auto params = *static_cast<ur_enqueue_usm_prefetch_params_t *>(pParams);
  urUSMPrefetchDirection = *(params.pflags);
  return UR_RESULT_SUCCESS;
}

TEST(USMPrefetchExp, CheckURCall) {
  using namespace sycl;
  unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urEnqueueUSMPrefetch",
                                            &redefinedEnqueueUSMPrefetch);
  queue q;
  int *Mem =
      (int *)malloc_shared(sizeof(int) * N, q.get_device(), q.get_context());

  // Check handler calls:
  q.submit([&](handler &cgh) {
    sycl::ext::oneapi::experimental::prefetch(cgh, Mem, sizeof(int) * N);
  });
  q.wait_and_throw();
  EXPECT_EQ(urUSMPrefetchDirection, UR_USM_MIGRATION_FLAG_HOST_TO_DEVICE);

  q.submit([&](handler &cgh) {
    sycl::ext::oneapi::experimental::prefetch(
        cgh, Mem, sizeof(int) * N,
        sycl::ext::oneapi::experimental::prefetch_type::device);
  });
  q.wait_and_throw();
  EXPECT_EQ(urUSMPrefetchDirection, UR_USM_MIGRATION_FLAG_HOST_TO_DEVICE);

  q.submit([&](handler &cgh) {
    sycl::ext::oneapi::experimental::prefetch(
        cgh, Mem, sizeof(int) * N,
        sycl::ext::oneapi::experimental::prefetch_type::host);
  });
  q.wait_and_throw();
  EXPECT_EQ(urUSMPrefetchDirection, UR_USM_MIGRATION_FLAG_DEVICE_TO_HOST);

  // Check queue calls:
  sycl::ext::oneapi::experimental::prefetch(q, Mem, sizeof(int) * N);
  q.wait_and_throw();
  EXPECT_EQ(urUSMPrefetchDirection, UR_USM_MIGRATION_FLAG_HOST_TO_DEVICE);

  sycl::ext::oneapi::experimental::prefetch(
      q, Mem, sizeof(int) * N,
      sycl::ext::oneapi::experimental::prefetch_type::device);
  q.wait_and_throw();
  EXPECT_EQ(urUSMPrefetchDirection, UR_USM_MIGRATION_FLAG_HOST_TO_DEVICE);

  sycl::ext::oneapi::experimental::prefetch(
      q, Mem, sizeof(int) * N,
      sycl::ext::oneapi::experimental::prefetch_type::host);
  q.wait_and_throw();
  EXPECT_EQ(urUSMPrefetchDirection, UR_USM_MIGRATION_FLAG_DEVICE_TO_HOST);

  free(Mem, q);
}
