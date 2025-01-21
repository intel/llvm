#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

TEST(AccessorHostTaskTest, ReadOnlyHostTask) {
  static constexpr sycl::access_mode acmode = sycl::access_mode::read;
  static constexpr sycl::access::target actarget =
      sycl::access::target::host_task;
  using mode_target_tag_type =
      std::add_const_t<sycl::mode_target_tag_t<acmode, actarget>>;

  using AccT = sycl::accessor<int, 1, acmode, actarget>;
  AccT acc;

  EXPECT_TRUE((std::is_same_v<mode_target_tag_type,
                              decltype(sycl::read_only_host_task)>));
}

TEST(AccessorHostTaskTest, ReadWriteHostTask) {
  static constexpr sycl::access_mode acmode = sycl::access_mode::read_write;
  static constexpr sycl::access::target actarget =
      sycl::access::target::host_task;
  using mode_target_tag_type =
      std::add_const_t<sycl::mode_target_tag_t<acmode, actarget>>;

  using AccT = sycl::accessor<int, 1, acmode, actarget>;
  AccT acc;

  EXPECT_TRUE((std::is_same_v<mode_target_tag_type,
                              decltype(sycl::read_write_host_task)>));
}

TEST(AccessorHostTaskTest, WriteOnlyHostTask) {
  static constexpr sycl::access_mode acmode = sycl::access_mode::write;
  static constexpr sycl::access::target actarget =
      sycl::access::target::host_task;
  using mode_target_tag_type =
      std::add_const_t<sycl::mode_target_tag_t<acmode, actarget>>;

  using AccT = sycl::accessor<int, 1, acmode, actarget>;
  AccT acc;

  EXPECT_TRUE((std::is_same_v<mode_target_tag_type,
                              decltype(sycl::write_only_host_task)>));
}