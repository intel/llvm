#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/queue.hpp>

using namespace sycl;

static bool InOrderFlagSeen = false;
ur_result_t urQueueCreateRedefineBefore(void *pParams) {
  auto params = *static_cast<ur_queue_create_params_t *>(pParams);
  EXPECT_TRUE(*params.ppProperties != nullptr);
  InOrderFlagSeen = !((*params.ppProperties)->flags &
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  return UR_RESULT_SUCCESS;
}

TEST(InOrderQueue, CheckFlagIsPassed) {
  unittest::UrMock<> Mock;
  platform Plt = sycl::platform();

  mock::getCallbacks().set_before_callback("urQueueCreate",
                                           &urQueueCreateRedefineBefore);

  EXPECT_FALSE(InOrderFlagSeen);
  queue q1{};
  EXPECT_FALSE(InOrderFlagSeen);
  queue q2{property::queue::in_order{}};
  EXPECT_TRUE(InOrderFlagSeen);
}
