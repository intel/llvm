#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/queue.hpp>

using namespace sycl;

static bool InOrderFlagSeen = false;
pi_result piextQueueCreateRedefineBefore(pi_context context, pi_device device,
                                         pi_queue_properties *properties,
                                         pi_queue *queue) {
  EXPECT_TRUE(properties != nullptr);
  EXPECT_TRUE(properties[0] == PI_QUEUE_FLAGS);
  InOrderFlagSeen =
      !(properties[1] & PI_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  return PI_SUCCESS;
}

TEST(InOrderQueue, CheckFlagIsPassed) {
  unittest::PiMock Mock;
  platform Plt = Mock.getPlatform();

  Mock.redefineBefore<detail::PiApiKind::piextQueueCreate>(
      piextQueueCreateRedefineBefore);

  EXPECT_FALSE(InOrderFlagSeen);
  queue q1{};
  EXPECT_FALSE(InOrderFlagSeen);
  queue q2{property::queue::in_order{}};
  EXPECT_TRUE(InOrderFlagSeen);
}
