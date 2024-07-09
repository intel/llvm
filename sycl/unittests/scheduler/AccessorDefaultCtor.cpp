#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"
#include <detail/handler_impl.hpp>

#include <helpers/PiMock.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/TestKernel.hpp>

#include <vector>

using namespace sycl;

TEST_F(SchedulerTest, AccDefaultCtorDoesntAffectDepGraph) {
  unittest::PiMock Mock;
  platform Plt = Mock.getPlatform();

  queue QueueDev(context(Plt), default_selector_v);
  MockScheduler MS;

  detail::QueueImplPtr QueueDevImpl = detail::getSyclObjImpl(QueueDev);

  std::vector<detail::Command *> ToEnqueue;

  MockHandlerCustomFinalize MockCGH(QueueDevImpl,
                                    /*CallerNeedsEvent=*/true);

  sycl::accessor<int, 0, sycl::access::mode::read_write, sycl::target::device>
      B;

  MockCGH.single_task<class acc_with_zero_dim>([=]() {
    int size = B.size();
    (void)size;
  });

  std::unique_ptr<sycl::detail::CG> CmdGroup = MockCGH.finalize();

  detail::Command *NewCmd = MS.addCG(std::move(CmdGroup), QueueDevImpl,
                                     ToEnqueue, /*EventNeeded=*/true);

  // if MDeps is empty, accessor built from default ctor does not affect
  // dependency graph in accordance with SYCL 2020
  EXPECT_TRUE(NewCmd->MDeps.empty());
}
