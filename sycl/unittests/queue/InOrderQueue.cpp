#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/queue.hpp>

using namespace sycl;

static bool InOrderFlagSeen = false;
pi_result piQueueCreateRedefineBefore(pi_context context, pi_device device,
                                      pi_queue_properties properties,
                                      pi_queue *queue) {
  InOrderFlagSeen = !(properties & PI_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  return PI_SUCCESS;
}

TEST(InOrderQueue, CheckFlagIsPassed) {
  unittest::PiMock Mock;
  platform Plt = Mock.getPlatform();

  Mock.redefineBefore<detail::PiApiKind::piQueueCreate>(
      piQueueCreateRedefineBefore);

  EXPECT_FALSE(InOrderFlagSeen);
  queue q1{};
  EXPECT_FALSE(InOrderFlagSeen);
  queue q2{property::queue::in_order{}};
  EXPECT_TRUE(InOrderFlagSeen);
}
