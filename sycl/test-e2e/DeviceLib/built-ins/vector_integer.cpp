// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

#include <array>
#include <cassert>

namespace s = sycl;

int main() {
  // max
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxSI2SI2>([=]() {
          AccR[0] = s::max(s::int2{5, 3}, s::int2{2, 7});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 5);
    assert(r2 == 7);
  }

  // max
  {
    s::uint2 r{0};
    {
      s::buffer<s::uint2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxUI2UI2>([=]() {
          AccR[0] = s::max(s::uint2{5, 3}, s::uint2{2, 7});
        });
      });
    }
    unsigned int r1 = r.x();
    unsigned int r2 = r.y();
    assert(r1 == 5);
    assert(r2 == 7);
  }

  // max
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxSI2SI1>([=]() {
          AccR[0] = s::max(s::int2{5, 3}, int{2});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 5);
    assert(r2 == 3);
  }

  // max (longlong2)
  {
    using longlong2 = s::vec<long long, 2>;
    longlong2 r{0};
    {
      s::buffer<longlong2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxSLL2SLL1>([=]() {
          AccR[0] = s::max(longlong2{5, 3}, 2ll);
        });
      });
    }
    long long r1 = r.x();
    long long r2 = r.y();
    assert(r1 == 5);
    assert(r2 == 3);
  }

  // max
  {
    s::uint2 r{0};
    {
      s::buffer<s::uint2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxUI2UI1>([=]() {
          AccR[0] = s::max(s::uint2{5, 3}, 2u);
        });
      });
    }
    unsigned int r1 = r.x();
    unsigned int r2 = r.y();
    assert(r1 == 5);
    assert(r2 == 3);
  }

  // max (ulonglong2)
  {
    using ulonglong2 = s::vec<unsigned long long, 2>;
    ulonglong2 r{0};
    {
      s::buffer<ulonglong2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxULL2ULL1>([=]() {
          AccR[0] = s::max(ulonglong2{5, 3}, 2ull);
        });
      });
    }
    unsigned long long r1 = r.x();
    unsigned long long r2 = r.y();
    assert(r1 == 5);
    assert(r2 == 3);
  }

  // min
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class minSI2SI2>([=]() {
          AccR[0] = s::min(s::int2{5, 3}, s::int2{2, 7});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 2);
    assert(r2 == 3);
  }

  // min
  {
    s::uint2 r{0};
    {
      s::buffer<s::uint2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class minUI2UI2>([=]() {
          AccR[0] = s::min(s::uint2{5, 3}, s::uint2{2, 7});
        });
      });
    }
    unsigned int r1 = r.x();
    unsigned int r2 = r.y();
    assert(r1 == 2);
    assert(r2 == 3);
  }

  // min
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class minSI2SI1>([=]() {
          AccR[0] = s::min(s::int2{5, 3}, int{2});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 2);
    assert(r2 == 2);
  }

  // min
  {
    s::uint2 r{0};
    {
      s::buffer<s::uint2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class minUI2UI1>([=]() {
          AccR[0] = s::min(s::uint2{5, 3}, 2u);
        });
      });
    }
    unsigned int r1 = r.x();
    unsigned int r2 = r.y();
    assert(r1 == 2);
    assert(r2 == 2);
  }

  // abs
  {
    s::uint2 r{0};
    {
      s::buffer<s::uint2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class absSI2>([=]() {
          AccR[0] = s::abs(s::int2{-5, -2});
        });
      });
    }
    unsigned int r1 = r.x();
    unsigned int r2 = r.y();
    assert(r1 == 5);
    assert(r2 == 2);
  }

  // abs (longlong)
  {
    using longlong2 = s::vec<long long, 2>;
    longlong2 r{0};
    {
      s::buffer<longlong2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class absSL2>([=]() {
          AccR[0] = s::abs(longlong2{-5, -2});
        });
      });
    }
    long long r1 = r.x();
    long long r2 = r.y();
    assert(r1 == 5);
    assert(r2 == 2);
  }

  // abs_diff
  {
    s::uint2 r{0};
    {
      s::buffer<s::uint2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class abs_diffSI2SI2>([=]() {
          AccR[0] = s::abs_diff(s::int2{-5, -2}, s::int2{-1, -1});
        });
      });
    }
    unsigned int r1 = r.x();
    unsigned int r2 = r.y();
    assert(r1 == 4);
    assert(r2 == 1);
  }

  // add_sat
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class add_satSI2SI2>([=]() {
          AccR[0] =
              s::add_sat(s::int2{0x7FFFFFFF, 0x7FFFFFFF}, s::int2{100, 90});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 0x7FFFFFFF);
    assert(r2 == 0x7FFFFFFF);
  }

  // hadd
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class haddSI2SI2>([=]() {
          AccR[0] = s::hadd(s::int2{0x0000007F, 0x0000007F},
                            s::int2{0x00000020, 0x00000020});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 0x0000004F);
    assert(r2 == 0x0000004F);
  }

  // rhadd
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class rhaddSI2SI2>([=]() {
          AccR[0] = s::rhadd(s::int2{0x0000007F, 0x0000007F},
                             s::int2{0x00000020, 0x00000020});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 0x00000050);
    assert(r2 == 0x00000050);
  }

  // clamp - 1
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class clampSI2SI2SI2>([=]() {
          AccR[0] = s::clamp(s::int2{5, 5}, s::int2{10, 10}, s::int2{30, 30});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 10);
    assert(r2 == 10);
  }

  // clamp - 2
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class clampSI2SI1SI1>([=]() {
          AccR[0] = s::clamp(s::int2{5, 5}, int{10}, int{30});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 10);
    assert(r2 == 10);
  }

  // clz
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class clzSI2>([=]() {
          AccR[0] = s::clz(s::int2{0x0FFFFFFF, 0x0FFFFFFF});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 4);
    assert(r2 == 4);
  }

  // ctz
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class ctzSI2>([=]() {
          AccR[0] = s::ctz(s::int2{0x7FFFFFF0, 0x7FFFFFF0});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 4);
    assert(r2 == 4);
  }

  // mad_hi
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mad_hiSI2SI2SI2>([=]() {
          AccR[0] = s::mad_hi(s::int2{0x10000000, 0x10000000},
                              s::int2{0x00000100, 0x00000100}, s::int2{1, 1});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 0x11);
    assert(r2 == 0x11);
  }

  // mad_sat
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mad_satSI2SI2SI2>([=]() {
          AccR[0] = s::mad_sat(s::int2{0x10000000, 0x10000000},
                               s::int2{0x00000100, 0x00000100}, s::int2{1, 1});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 0x7FFFFFFF);
    assert(r2 == 0x7FFFFFFF);
  }

  // mul_hi
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mul_hiSI2SI2>([=]() {
          AccR[0] = s::mul_hi(s::int2{0x10000000, 0x10000000},
                              s::int2{0x00000100, 0x00000100});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 0x10);
    assert(r2 == 0x10);
  }

  // rotate
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class rotateSI2SI2>([=]() {
          AccR[0] = s::rotate(s::int2{0x11100000, 0x11100000}, s::int2{12, 12});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 0x00000111);
    assert(r2 == 0x00000111);
  }

  // sub_sat
  {
    auto TestSubSat = [](s::int2 x, s::int2 y) {
      s::int2 r{0};
      {
        s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
        s::queue myQueue;
        myQueue.submit([&](s::handler &cgh) {
          auto AccR = BufR.get_access<s::access::mode::write>(cgh);
          cgh.single_task<class sub_satSI2SI2>(
              [=]() { AccR[0] = s::sub_sat(x, y); });
        });
      }
      return r;
    };
    s::int2 r1 = TestSubSat(s::int2{10, 10}, s::int2{0x80000000, 0x80000000});
    int r1x = r1.x();
    int r1y = r1.y();
    assert(r1x == 0x7FFFFFFF);
    assert(r1y == 0x7FFFFFFF);
    s::int2 r2 = TestSubSat(s::int2{0x7FFFFFFF, 0x80000000},
                            s::int2{0xFFFFFFFF, 0x00000001});
    int r2x = r2.x();
    int r2y = r2.y();
    assert(r2x == 0x7FFFFFFF);
    assert(r2y == 0x80000000);
    s::int2 r3 = TestSubSat(s::int2{10499, 30678}, s::int2{30678, 10499});
    int r3x = r3.x();
    int r3y = r3.y();
    assert(r3x == -20179);
    assert(r3y == 20179);
  }

  // upsample - 1
  {
    s::ushort2 r{0};
    {
      s::buffer<s::ushort2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleUC2UC2>([=]() {
          AccR[0] = s::upsample(s::uchar2{0x10, 0x10}, s::uchar2{0x10, 0x10});
        });
      });
    }
    unsigned short r1 = r.x();
    unsigned short r2 = r.y();
    assert(r1 == 0x1010);
    assert(r2 == 0x1010);
  }

  // upsample - 2
  {
    s::short2 r{0};
    {
      s::buffer<s::short2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleSC2UC2>([=]() {
          AccR[0] = s::upsample(s::char2{0x10, 0x10}, s::uchar2{0x10, 0x10});
        });
      });
    }
    short r1 = r.x();
    short r2 = r.y();
    assert(r1 == 0x1010);
    assert(r2 == 0x1010);
  }

  // upsample - 3
  {
    s::uint2 r{0};
    {
      s::buffer<s::uint2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleUS2US2>([=]() {
          AccR[0] = s::upsample(s::ushort2{0x0010, 0x0010},
                                s::ushort2{0x0010, 0x0010});
        });
      });
    }
    unsigned int r1 = r.x();
    unsigned int r2 = r.y();
    assert(r1 == 0x00100010);
    assert(r2 == 0x00100010);
  }

  // upsample - 4
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleSS2US2>([=]() {
          AccR[0] = s::upsample(s::short2{0x0010, 0x0010},
                                s::ushort2{0x0010, 0x0010});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 0x00100010);
    assert(r2 == 0x00100010);
  }

  // upsample - 5
  {
    s::ulong2 r{0};
    {
      s::buffer<s::ulong2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleUI2UI2>([=]() {
          AccR[0] = s::upsample(s::uint2{0x00000010, 0x00000010},
                                s::uint2{0x00000010, 0x00000010});
        });
      });
    }
    unsigned long long r1 = r.x();
    unsigned long long r2 = r.y();
    assert(r1 == 0x0000001000000010);
    assert(r2 == 0x0000001000000010);
  }

  // upsample - 6
  {
    s::long2 r{0};
    {
      s::buffer<s::long2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleSI2UI2>([=]() {
          AccR[0] = s::upsample(s::int2{0x00000010, 0x00000010},
                                s::uint2{0x00000010, 0x00000010});
        });
      });
    }
    long long r1 = r.x();
    long long r2 = r.y();
    assert(r1 == 0x0000001000000010);
    assert(r2 == 0x0000001000000010);
  }

  // popcount
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class popcountSI2>([=]() {
          AccR[0] = s::popcount(s::int2{0x000000FF, 0x000000FF});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 8);
    assert(r2 == 8);
  }

  // mad24
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mad24SI2SI2SI2>([=]() {
          AccR[0] = s::mad24(s::int2{0xFFFFFFFF, 0xFFFFFFFF}, s::int2{20, 20},
                             s::int2{20, 20});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == 0);
    assert(r2 == 0);
  }

  // mul24
  {
    s::int2 r{0};
    {
      s::buffer<s::int2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mul24SI2SI2SI2>([=]() {
          AccR[0] = s::mul24(s::int2{0xFFFFFFFF, 0xFFFFFFFF}, s::int2{20, 20});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    assert(r1 == -20);
    assert(r2 == -20);
  }

  return 0;
}
