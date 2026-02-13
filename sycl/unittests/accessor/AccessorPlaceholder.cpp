#include <gtest/gtest.h>

#include <helpers/UrMock.hpp>
#include <numeric>
#include <sycl/sycl.hpp>

TEST(AccessorPlaceholderTest, NoCommandGroupPlaceholderNoneTargetDevice) {
  static constexpr sycl::access_mode acmode = sycl::access_mode::read;
  static constexpr sycl::access::target actarget = sycl::access::target::device;
  using AccT = sycl::accessor<int, 1, acmode, actarget>;
  int data(14);
  sycl::range<1> r(1);
  sycl::buffer<int, 1> data_buf(&data, r);
  AccT acc(data_buf);
  EXPECT_TRUE(acc.is_placeholder());
}

TEST(AccessorPlaceholderTest, NoCommandGroupPlaceholderTrueTargetDevice) {
  static constexpr sycl::access_mode acmode = sycl::access_mode::read;
  static constexpr sycl::access::target actarget = sycl::access::target::device;
  static constexpr sycl::access::placeholder acplaceholder =
      sycl::access::placeholder::true_t;
  using AccT = sycl::accessor<int, 1, acmode, actarget, acplaceholder>;
  int data(14);
  sycl::range<1> r(1);
  sycl::buffer<int, 1> data_buf(&data, r);
  AccT acc(data_buf);
  EXPECT_TRUE(acc.is_placeholder());
}

TEST(AccessorPlaceholderTest, NoCommandGroupPlaceholderFalseTargetDevice) {
  static constexpr sycl::access_mode acmode = sycl::access_mode::read;
  static constexpr sycl::access::target actarget = sycl::access::target::device;
  static constexpr sycl::access::placeholder acplaceholder =
      sycl::access::placeholder::false_t;
  using AccT = sycl::accessor<int, 1, acmode, actarget, acplaceholder>;
  int data(14);
  sycl::range<1> r(1);
  sycl::buffer<int, 1> data_buf(&data, r);
  AccT acc(data_buf);
  EXPECT_TRUE(acc.is_placeholder());
}

TEST(AccessorPlaceholderTest, PlaceholderNoneTargetDevice) {
  static constexpr sycl::access_mode acmode = sycl::access_mode::read;
  static constexpr sycl::access::target actarget = sycl::access::target::device;
  using AccT = sycl::accessor<int, 1, acmode, actarget>;
  int data(14);
  sycl::range<1> r(1);
  sycl::buffer<int, 1> data_buf(&data, r);
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  sycl::queue q{Plt.get_devices()[0]};
  q.submit([&](sycl::handler &cgh) {
    AccT acc(data_buf, cgh);
    EXPECT_FALSE(acc.is_placeholder());
  });
}

TEST(AccessorPlaceholderTest, PlaceholderTrueTargetDevice) {
  static constexpr sycl::access_mode acmode = sycl::access_mode::read;
  static constexpr sycl::access::target actarget = sycl::access::target::device;
  static constexpr sycl::access::placeholder acplaceholder =
      sycl::access::placeholder::true_t;
  using AccT = sycl::accessor<int, 1, acmode, actarget, acplaceholder>;
  int data(14);
  sycl::range<1> r(1);
  sycl::buffer<int, 1> data_buf(&data, r);
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  sycl::queue q{Plt.get_devices()[0]};
  q.submit([&](sycl::handler &cgh) {
    AccT acc(data_buf, cgh);
    EXPECT_FALSE(acc.is_placeholder());
  });
}

TEST(AccessorPlaceholderTest, PlaceholderFalseTargetDevice) {
  static constexpr sycl::access_mode acmode = sycl::access_mode::read;
  static constexpr sycl::access::target actarget = sycl::access::target::device;
  static constexpr sycl::access::placeholder acplaceholder =
      sycl::access::placeholder::false_t;
  using AccT = sycl::accessor<int, 1, acmode, actarget, acplaceholder>;
  int data(14);
  sycl::range<1> r(1);
  sycl::buffer<int, 1> data_buf(&data, r);
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  sycl::queue q{Plt.get_devices()[0]};
  q.submit([&](sycl::handler &cgh) {
    AccT acc(data_buf, cgh);
    EXPECT_FALSE(acc.is_placeholder());
  });
}

TEST(AccessorPlaceholderTest, PlaceholderNoneTargetHost) {
  static constexpr sycl::access_mode acmode = sycl::access_mode::read;
  static constexpr sycl::access::target actarget =
      sycl::access::target::host_buffer;
  using AccT = sycl::accessor<int, 1, acmode, actarget>;
  int data(14);
  sycl::range<1> r(1);
  sycl::buffer<int, 1> data_buf(&data, r);
  AccT acc(data_buf);
  EXPECT_FALSE(acc.is_placeholder());
  EXPECT_EQ(acc[0], data);
}

TEST(AccessorPlaceholderTest, PlaceholderTrueTargetHost) {
  static constexpr sycl::access_mode acmode = sycl::access_mode::read;
  static constexpr sycl::access::target actarget =
      sycl::access::target::host_buffer;
  static constexpr sycl::access::placeholder acplaceholder =
      sycl::access::placeholder::true_t;
  using AccT = sycl::accessor<int, 1, acmode, actarget, acplaceholder>;
  int data(14);
  sycl::range<1> r(1);
  sycl::buffer<int, 1> data_buf(&data, r);
  AccT acc(data_buf);
  EXPECT_FALSE(acc.is_placeholder());
}

TEST(AccessorPlaceholderTest, PlaceholderFalseTargetHost) {
  static constexpr sycl::access_mode acmode = sycl::access_mode::read;
  static constexpr sycl::access::target actarget =
      sycl::access::target::host_buffer;
  static constexpr sycl::access::placeholder acplaceholder =
      sycl::access::placeholder::false_t;
  using AccT = sycl::accessor<int, 1, acmode, actarget, acplaceholder>;
  int data(14);
  sycl::range<1> r(1);
  sycl::buffer<int, 1> data_buf(&data, r);
  AccT acc(data_buf);
  EXPECT_FALSE(acc.is_placeholder());
}
