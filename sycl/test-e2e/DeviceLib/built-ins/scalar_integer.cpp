// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}

#include <sycl/sycl.hpp>

#include <array>
#include <cassert>

namespace s = sycl;

int main() {
  // max
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxSI1SI1>([=]() { AccR[0] = s::max(5, 2); });
      });
    }
    assert(r == 5);
  }

  // max
  {
    unsigned int r{0};
    {
      s::buffer<unsigned int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxUI1UI1>([=]() { AccR[0] = s::max(5u, 2u); });
      });
    }
    assert(r == 5);
  }

  // min
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class minSI1SI1>([=]() { AccR[0] = s::min(5, 2); });
      });
    }
    assert(r == 2);
  }

  // min (long long)
  {
    long long r{0};
    {
      s::buffer<long long, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class minSLL1SLL1>(
            [=]() { AccR[0] = s::min(5ll, 2ll); });
      });
    }
    assert(r == 2);
  }

  // min
  {
    unsigned int r{0};
    {
      s::buffer<unsigned int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class minUI1UI1>([=]() { AccR[0] = s::min(5u, 2u); });
      });
    }
    assert(r == 2);
  }

  // min (unsigned long long)
  {
    unsigned long long r{0};
    {
      s::buffer<unsigned long long, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class minULL1ULL1>(
            [=]() { AccR[0] = s::min(5ull, 2ull); });
      });
    }
    assert(r == 2);
  }

  // abs
  {
    unsigned int r{0};
    {
      s::buffer<unsigned int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class absSI1>([=]() { AccR[0] = s::abs(-5); });
      });
    }
    assert(r == 5);
  }

  // abs_diff
  {
    unsigned int r{0};
    {
      s::buffer<unsigned int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class abs_diffSI1SI1>(
            [=]() { AccR[0] = s::abs_diff(-5, -1); });
      });
    }
    assert(r == 4);
  }

  // abs_diff(uchar)
  {
    unsigned char r{0};
    {
      s::buffer<unsigned char, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class abs_diffUC1UC1>([=]() {
          AccR[0] = s::abs_diff((unsigned char)3, (unsigned char)250);
        });
      });
    }
    assert(r == 247);
  }

  // add_sat
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class add_satSI1SI1>(
            [=]() { AccR[0] = s::add_sat(0x7FFFFFFF, 100); });
      });
    }
    assert(r == 0x7FFFFFFF);
  }

  // hadd
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class haddSI1SI1>(
            [=]() { AccR[0] = s::hadd(0x0000007F, 0x00000020); });
      });
    }
    assert(r == 0x0000004F);
  }

  // rhadd
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class rhaddSI1SI1>(
            [=]() { AccR[0] = s::rhadd(0x0000007F, 0x00000020); });
      });
    }
    assert(r == 0x50);
  }

  // clamp
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class clampSI1SI1SI1>(
            [=]() { AccR[0] = s::clamp(5, 10, 30); });
      });
    }
    assert(r == 10);
  }

  // clz
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class clzSI1>([=]() { AccR[0] = s::clz(0x0FFFFFFF); });
      });
    }
    assert(r == 4);
  }

  // ctz
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class ctzSI1>(
            [=]() { AccR[0] = s::ctz(0x7FFFFFF0); });
      });
    }
    assert(r == 4);
  }

  // mad_hi
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mad_hiSI1SI1SI1>([=]() {
          AccR[0] = s::mad_hi(0x10000000, 0x00000100, 0x00000001);
        }); // 2^28 * 2^8 = 2^36 -> 0x10 00000000.
      });
    }
    assert(r == 0x11);
  }

  // mad_sat
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mad_satSI1SI1SI1>([=]() {
          AccR[0] = s::mad_sat(0x10000000, 0x00000100, 0x00000001);
        }); // 2^31 * 2^8 = 2^39 -> 0x80 00000000 -> reuslt is saturated in the
            // product.
      });
    }
    assert(r == 0x7FFFFFFF);
  }

  // mad_sat test two
  {
    char r(0);
    char exp(120);
    {
      sycl::buffer<char, 1> buf(&r, sycl::range<1>(1));
      sycl::queue q;
      q.submit([&](sycl::handler &cgh) {
        auto acc = buf.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class kernel>([=]() {
          signed char inputData_0(-17);
          signed char inputData_1(-10);
          signed char inputData_2(-50);
          acc[0] = sycl::mad_sat(inputData_0, inputData_1, inputData_2);
        });
      });
    }
    assert(r == exp); // Should return the real number of i0*i1+i2 in CPU
                      // Only fails in vector, but passes in scalar.
  }

  // mul_hi
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mul_hiSI1SI1>([=]() {
          AccR[0] = s::mul_hi(0x10000000, 0x00000100);
        }); // 2^28 * 2^8 = 2^36 -> 0x10 00000000.
      });
    }
    assert(r == 0x10);
  }

  // mul_hi with negative result w/ carry
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mul_hiSI1SI2>([=]() {
          AccR[0] = s::mul_hi(-0x10000000, 0x00000100);
        }); // -2^28 * 2^8 = -2^36 -> -0x10 (FFFFFFF0) 00000000.
      });
    }
    assert(r == -0x10);
  }

  // mul_hi with negative result w/o carry
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mul_hiSI1SI3>([=]() {
          AccR[0] = s::mul_hi(-0x10000000, 0x00000101);
        }); // -2^28 * (2^8 + 1) = -2^36 - 2^28 -> -0x11 (FFFFFFEF) -0x10000000
            // (F0000000).
      });
    }
    assert(r == -0x11);
  }

  // rotate
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class rotateSI1SI1>(
            [=]() { AccR[0] = s::rotate(0x11100000, 12); });
      });
    }
    assert(r == 0x00000111);
  }

  // rotate (with large rotate size)
  {
    char r{0};
    {
      s::buffer<char, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class rotateSI1SI2>(
            [=]() { AccR[0] = s::rotate((char)0xe0, (char)50); });
      });
    }
    assert((unsigned char)r == 0x83);
  }
  // sub_sat
  {
    auto TestSubSat = [](int x, int y) {
      int r{0};
      {
        s::buffer<int, 1> BufR(&r, s::range<1>(1));
        s::queue myQueue;
        myQueue.submit([&](s::handler &cgh) {
          auto AccR = BufR.get_access<s::access::mode::write>(cgh);
          cgh.single_task<class sub_satSI1SI1>(
              [=]() { AccR[0] = s::sub_sat(x, y); });
        });
      }
      return r;
    };
    // 10 - (-2^31(minimum value)) = saturates on Maximum value
    int r1 = TestSubSat(10, 0x80000000);
    assert(r1 == 0x7FFFFFFF);
    int r2 = TestSubSat(0x7FFFFFFF, 0xFFFFFFFF);
    assert(r2 == 0x7FFFFFFF);
    int r3 = TestSubSat(0x80000000, 0x00000001);
    assert(r3 == 0x80000000);
    int r4 = TestSubSat(10499, 30678);
    assert(r4 == -20179);
  }

  // upsample - 1
  {
    unsigned short r{0};
    {
      s::buffer<unsigned short, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleUC1UC1>([=]() {
          AccR[0] = s::upsample((unsigned char)0x10, (unsigned char)0x10);
        });
      });
    }
    assert(r == 0x1010);
  }

  // upsample - 2
  {
    short r{0};
    {
      s::buffer<short, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleSC1UC1>(
            [=]() { AccR[0] = s::upsample((int8_t)0x10, (uint8_t)0x10); });
      });
    }
    assert(r == 0x1010);
  }

  // upsample - 3
  {
    unsigned int r{0};
    {
      s::buffer<unsigned int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleUS1US1>([=]() {
          AccR[0] = s::upsample((unsigned short)0x0010, (unsigned short)0x0010);
        });
      });
    }
    assert(r == 0x00100010);
  }

  // upsample - 4
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleSS1US1>([=]() {
          AccR[0] = s::upsample((short)0x0010, (unsigned short)0x0010);
        });
      });
    }
    assert(r == 0x00100010);
  }

  // upsample - 5
  {
    unsigned long long r{0};
    {
      s::buffer<unsigned long long, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleUI1UI1>(
            [=]() { AccR[0] = s::upsample(0x00000010u, 0x00000010u); });
      });
    }
    assert(r == 0x0000001000000010);
  }

  // upsample - 6
  {
    long long r{0};
    {
      s::buffer<long long, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleSI1UI1>(
            [=]() { AccR[0] = s::upsample(0x00000010, 0x00000010u); });
      });
    }
    assert(r == 0x0000001000000010);
  }

  // popcount
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class popcountSI1>(
            [=]() { AccR[0] = s::popcount(0x000000FF); });
      });
    }
    assert(r == 8);
  }

  // mad24
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mad24SI1SI1SI1>(
            [=]() { AccR[0] = s::mad24((int)0xFFFFFFFF, (int)20, (int)20); });
      });
    }
    assert(r == 0);
  }

  // mul24
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mul24SI1SI1>(
            [=]() { AccR[0] = s::mul24((int)0xFFFFFFFF, (int)20); });
      });
    }
    assert(r == -20);
  }

  return 0;
}
