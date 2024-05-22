#include <detail/queue_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>
#include <sycl/queue.hpp>

using namespace sycl;

// Checks that the queue hash uses its unique ID.
TEST(QueueHash, QueueHashUsesID) {
  unittest::PiMock Mock;
  queue Q;
  unsigned long long ID = detail::getSyclObjImpl(Q)->getQueueID();
  ASSERT_EQ(std::hash<unsigned long long>{}(ID), std::hash<queue>{}(Q));
}
