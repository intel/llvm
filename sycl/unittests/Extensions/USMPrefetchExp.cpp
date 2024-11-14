#include <helpers/UrMock.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/USM/prefetch_exp.hpp>
#include <sycl/usm.hpp>

#include <gtest/gtest.h>

static constexpr int N = 8;
static ur_usm_migration_flags_t urUSMPrefetchDirection = -1;

// TODO: FIGURE OUT WHEN COMMANDBUF GETS CALLED AND IMPLEMENT COMMANDBUF TESTING

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
  int *mem =
      (int *)malloc_shared(sizeof(int) * N, q.get_device(), q.get_context());

  // Check handler calls:
  q.submit(
      [&](handler &cgh) { cgh.ext_oneapi_prefetch_exp(mem, sizeof(int) * N); });
  q.wait_and_throw();
  EXPECT_EQ(urUSMPrefetchDirection, UR_USM_MIGRATION_FLAG_HOST_TO_DEVICE);

  q.submit([&](handler &cgh) {
    cgh.ext_oneapi_prefetch_exp(
        mem, sizeof(int) * N,
        sycl::ext::oneapi::experimental::migration_direction::HOST_TO_DEVICE);
  });
  q.wait_and_throw();
  EXPECT_EQ(urUSMPrefetchDirection, UR_USM_MIGRATION_FLAG_HOST_TO_DEVICE);

  q.submit([&](handler &cgh) {
    cgh.ext_oneapi_prefetch_exp(
        mem, sizeof(int) * N,
        sycl::ext::oneapi::experimental::migration_direction::DEVICE_TO_HOST);
  });
  q.wait_and_throw();
  EXPECT_EQ(urUSMPrefetchDirection, UR_USM_MIGRATION_FLAG_DEVICE_TO_HOST);

  // Check queue calls:
  q.ext_oneapi_prefetch_exp(mem, sizeof(int) * N);
  q.wait_and_throw();
  EXPECT_EQ(urUSMPrefetchDirection, UR_USM_MIGRATION_FLAG_HOST_TO_DEVICE);

  q.ext_oneapi_prefetch_exp(
      mem, sizeof(int) * N,
      sycl::ext::oneapi::experimental::migration_direction::HOST_TO_DEVICE);
  q.wait_and_throw();
  EXPECT_EQ(urUSMPrefetchDirection, UR_USM_MIGRATION_FLAG_HOST_TO_DEVICE);

  q.ext_oneapi_prefetch_exp(
      mem, sizeof(int) * N,
      sycl::ext::oneapi::experimental::migration_direction::DEVICE_TO_HOST);
  q.wait_and_throw();
  EXPECT_EQ(urUSMPrefetchDirection, UR_USM_MIGRATION_FLAG_DEVICE_TO_HOST);

  // TODO: not sure what else to test for, check event? I don't think there's
  // any other parameters to validate....
}
