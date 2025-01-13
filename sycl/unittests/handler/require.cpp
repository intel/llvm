#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>

#include <sycl/sycl.hpp>

TEST(Require, RequireWithNonPlaceholderAccessor) {
  sycl::unittest::UrMock<> Mock;
  sycl::queue Q;
  int data = 5;
  {
    sycl::buffer<int, 1> buf(&data, 1);
    Q.submit([&](sycl::handler &h) {
      auto acc = buf.get_access<sycl::access::mode::read_write>(h);
      // It should be compilable and does nothing according to the spec
      h.require(acc);
    });
    Q.wait();
  }
}

TEST(Require, checkIfAccBoundedToHandler) {
  std::string msg("placeholder accessor must be bound by calling "
                  "handler::require() before it can be used.");
  sycl::unittest::UrMock<> Mock;
  sycl::queue Q;
  int data = 0;

  // Check fill() command
  {
    try {
      sycl::buffer<int, 1> buf(&data, 1);
      sycl::accessor acc(buf);
      Q.submit([&](sycl::handler &h) { h.fill(acc, 1); });
      Q.wait();
    } catch (sycl::exception const &e) {
      EXPECT_EQ(e.what(), msg);
    }

    // Should pass without any exception thrown
    {
      sycl::buffer<int, 1> buf(&data, 1);
      sycl::accessor acc(buf);
      Q.submit([&](sycl::handler &h) {
        h.require(acc);
        h.fill(acc, 1);
      });
      Q.wait();
    }
  }

  // Check update_host() command
  {
    int data = 0;
    try {
      sycl::buffer<int, 1> buf(&data, 1);
      sycl::accessor acc(buf);
      Q.submit([&](sycl::handler &h) { h.update_host(acc); });
      Q.wait();
    } catch (sycl::exception const &e) {
      EXPECT_EQ(e.what(), msg);
    }

    // Should pass without any exception thrown
    {
      sycl::buffer<int, 1> buf(&data, 1);
      sycl::accessor acc(buf);
      Q.submit([&](sycl::handler &h) {
        h.require(acc);
        h.update_host(acc);
      });
      Q.wait();
    }
  }

  // Check different copy() variants
  {
    // void copy(accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsPlaceholder> src,
    //           std::shared_ptr<DestT> dest)
    {
      std::shared_ptr<int> ptr(new int(0));
      try {
        sycl::buffer<int, 1> buf(&data, 1);
        sycl::accessor acc(buf);
        Q.submit([&](sycl::handler &h) { h.copy(acc, ptr); });
        Q.wait();
      } catch (sycl::exception const &e) {
        EXPECT_EQ(e.what(), msg);
      }

      // Should pass without any exception thrown
      {
        sycl::buffer<int, 1> buf(&data, 1);
        sycl::accessor acc(buf);
        Q.submit([&](sycl::handler &h) {
          h.require(acc);
          h.copy(acc, ptr);
        });
        Q.wait();
      }
    }

    // void copy(std::shared_ptr<SrcT> src,
    //           accessor<DestT, DestDims, DestMode, DestTgt, IsPlaceholder>
    //           dest)
    {
      std::shared_ptr<int> ptr(new int(0));
      try {
        sycl::buffer<int, 1> buf(&data, 1);
        sycl::accessor acc(buf);
        Q.submit([&](sycl::handler &h) { h.copy(ptr, acc); });
        Q.wait();
      } catch (sycl::exception const &e) {
        EXPECT_EQ(e.what(), msg);
      }

      // Should pass without any exception thrown
      {
        sycl::buffer<int, 1> buf(&data, 1);
        sycl::accessor acc(buf);
        Q.submit([&](sycl::handler &h) {
          h.require(acc);
          h.copy(ptr, acc);
        });
        Q.wait();
      }
    }

    // void copy(accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsPlaceholder> src,
    //           DestT* dest)
    {
      int *ptr = new int(0);
      try {
        sycl::buffer<int, 1> buf(&data, 1);
        sycl::accessor acc(buf);
        Q.submit([&](sycl::handler &h) { h.copy(acc, ptr); });
        Q.wait();
      } catch (sycl::exception const &e) {
        EXPECT_EQ(e.what(), msg);
      }

      // Should pass without any exception thrown
      {
        sycl::buffer<int, 1> buf(&data, 1);
        sycl::accessor acc(buf);
        Q.submit([&](sycl::handler &h) {
          h.require(acc);
          h.copy(acc, ptr);
        });
        Q.wait();
      }
    }

    // void copy(const SrcT* src,
    //           accessor<DestT, DestDims, DestMode, DestTgt, IsPlaceholder>
    //           dest)
    {
      int *ptr = new int(0);
      try {
        sycl::buffer<int, 1> buf(&data, 1);
        sycl::accessor acc(buf);
        Q.submit([&](sycl::handler &h) { h.copy(ptr, acc); });
        Q.wait();
      } catch (sycl::exception const &e) {
        EXPECT_EQ(e.what(), msg);
      }

      // Should pass without any exception thrown
      {
        sycl::buffer<int, 1> buf(&data, 1);
        sycl::accessor acc(buf);
        Q.submit([&](sycl::handler &h) {
          h.require(acc);
          h.copy(ptr, acc);
        });
        Q.wait();
      }
    }

    // void copy(accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsSrcPlaceholder> src,
    //           accessor<DestT, DestDims, DestMode, DestTgt, IsDestPlaceholder>
    //           dest)
    {
      int data2 = 0;
      try {
        sycl::buffer<int, 1> buf(&data, 1);
        sycl::buffer<int, 1> buf2(&data2, 1);
        sycl::accessor acc(buf);
        sycl::accessor acc2(buf2);
        Q.submit([&](sycl::handler &h) { h.copy(acc, acc2); });
        Q.wait();
      } catch (sycl::exception const &e) {
        EXPECT_EQ(e.what(), msg);
      }
      try {
        sycl::buffer<int, 1> buf(&data, 1);
        sycl::buffer<int, 1> buf2(&data2, 1);
        sycl::accessor acc(buf);
        sycl::accessor acc2(buf2);
        Q.submit([&](sycl::handler &h) {
          h.require(acc);
          h.copy(acc, acc2);
        });
        Q.wait();
      } catch (sycl::exception const &e) {
        EXPECT_EQ(e.what(), msg);
      }
      try {
        sycl::buffer<int, 1> buf(&data, 1);
        sycl::buffer<int, 1> buf2(&data2, 1);
        sycl::accessor acc(buf);
        sycl::accessor acc2(buf2);
        Q.submit([&](sycl::handler &h) {
          h.require(acc2);
          h.copy(acc, acc2);
        });
        Q.wait();
      } catch (sycl::exception const &e) {
        EXPECT_EQ(e.what(), msg);
      }

      // Should pass without any exception thrown
      {
        sycl::buffer<int, 1> buf(&data, 1);
        sycl::buffer<int, 1> buf2(&data2, 1);
        sycl::accessor acc(buf);
        sycl::accessor acc2(buf2);
        Q.submit([&](sycl::handler &h) {
          h.require(acc);
          h.require(acc2);
          h.copy(acc, acc2);
        });
        Q.wait();
      }
    }
  }
}
