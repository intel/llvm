// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// TODO: ptxas fatal   : Unresolved extern function '_Z17__spirv_ocl_s_maxii'
// XFAIL: cuda

#include <CL/sycl.hpp>

#include <array>
#include <cassert>

namespace s = cl::sycl;

int main() {
  // max
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxSI1SI1>([=]() {
          AccR[0] = s::max(s::cl_int{ 5 }, s::cl_int{ 2 });
        });
      });
    }
    assert(r == 5);
  }

  // max
  {
    s::cl_uint r{ 0 };
    {
      s::buffer<s::cl_uint, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxUI1UI1>([=]() {
          AccR[0] = s::max(s::cl_uint{ 5 }, s::cl_uint{ 2 });
        });
      });
    }
    assert(r == 5);
  }

  // min
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class minSI1SI1>([=]() {
          AccR[0] = s::min(s::cl_int{ 5 }, s::cl_int{ 2 });
        });
      });
    }
    assert(r == 2);
  }

  // min (longlong)
  {
    s::longlong r{ 0 };
    {
      s::buffer<s::longlong, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class minSLL1SLL1>([=]() {
          AccR[0] = s::min(s::longlong{ 5 }, s::longlong{ 2 });
        });
      });
    }
    assert(r == 2);
  }

  // min
  {
    s::cl_uint r{ 0 };
    {
      s::buffer<s::cl_uint, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class minUI1UI1>([=]() {
          AccR[0] = s::min(s::cl_uint{ 5 }, s::cl_uint{ 2 });
        });
      });
    }
    assert(r == 2);
  }

  // min (ulonglong)
  {
    s::ulonglong r{ 0 };
    {
      s::buffer<s::ulonglong, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class minULL1ULL1>([=]() {
          AccR[0] = s::min(s::ulonglong{ 5 }, s::ulonglong{ 2 });
        });
      });
    }
    assert(r == 2);
  }

  // abs
  {
    s::cl_uint r{ 0 };
    {
      s::buffer<s::cl_uint, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class absSI1>([=]() {
          AccR[0] = s::abs(s::cl_int{ -5 });
        });
      });
    }
    assert(r == 5);
  }

  // abs_diff
  {
    s::cl_uint r{ 0 };
    {
      s::buffer<s::cl_uint, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class abs_diffSI1SI1>([=]() {
          AccR[0] = s::abs_diff(s::cl_int{ -5 }, s::cl_int{ -1 });
        });
      });
    }
    assert(r == 4);
  }

  // abs_diff(uchar)
  {
    s::cl_uchar r{ 0 };
    {
      s::buffer<s::cl_uchar, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class abs_diffUC1UC1>([=]() {
          AccR[0] = s::abs_diff(s::uchar{ 3 }, s::uchar{ 250 });
        });
      });
    }
    assert(r == 247);
  }

  // add_sat
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class add_satSI1SI1>([=]() {
          AccR[0] = s::add_sat(s::cl_int{ 0x7FFFFFFF }, s::cl_int{ 100 });
        });
      });
    }
    assert(r == 0x7FFFFFFF);
  }

  // hadd
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class haddSI1SI1>([=]() {
          AccR[0] = s::hadd(s::cl_int{ 0x0000007F }, s::cl_int{ 0x00000020 });
        });
      });
    }
    assert(r == 0x0000004F);
  }

  // rhadd
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class rhaddSI1SI1>([=]() {
          AccR[0] = s::rhadd(s::cl_int{ 0x0000007F }, s::cl_int{ 0x00000020 });
        });
      });
    }
    assert(r == 0x50);
  }

  // clamp
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class clampSI1SI1SI1>([=]() {
          AccR[0] = s::clamp(s::cl_int{ 5 }, s::cl_int{ 10 }, s::cl_int{ 30 });
        });
      });
    }
    assert(r == 10);
  }

  // clz
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class clzSI1>([=]() {
          AccR[0] = s::clz(s::cl_int{ 0x0FFFFFFF });
        });
      });
    }
    assert(r == 4);
  }

  // ctz
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class ctzSI1>([=]() {
          AccR[0] = s::intel::ctz(s::cl_int{ 0x7FFFFFF0 });
        });
      });
    }
    assert(r == 4);
  }

  // mad_hi
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mad_hiSI1SI1SI1>([=]() {
          AccR[0] = s::mad_hi(s::cl_int{ 0x10000000 }, s::cl_int{ 0x00000100 },
                              s::cl_int{ 0x00000001 });
        }); // 2^28 * 2^8 = 2^36 -> 0x10 00000000.
      });
    }
    assert(r == 0x11);
  }

  // mad_sat
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mad_satSI1SI1SI1>([=]() {
          AccR[0] = s::mad_sat(s::cl_int{ 0x10000000 }, s::cl_int{ 0x00000100 },
                               s::cl_int{ 0x00000001 });
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
      cl::sycl::buffer<char, 1> buf(&r, cl::sycl::range<1>(1));
      cl::sycl::queue q;
      q.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
        cgh.single_task<class kernel>([=]() {
          signed char inputData_0(-17);
          signed char inputData_1(-10);
          signed char inputData_2(-50);
          acc[0] = cl::sycl::mad_sat(inputData_0, inputData_1, inputData_2);
        });
      });
    }
    assert(r == exp); // Should return the real number of i0*i1+i2 in CPU
                              // Only fails in vector, but passes in scalar.

  }

  // mul_hi
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mul_hiSI1SI1>([=]() {
          AccR[0] = s::mul_hi(s::cl_int{ 0x10000000 }, s::cl_int{ 0x00000100 });
        }); // 2^28 * 2^8 = 2^36 -> 0x10 00000000.
      });
    }
    assert(r == 0x10);
  }

  // mul_hi with negative result w/ carry
  {
    s::cl_int r{0};
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mul_hiSI1SI2>([=]() {
          AccR[0] = s::mul_hi(s::cl_int{-0x10000000}, s::cl_int{0x00000100});
        }); // -2^28 * 2^8 = -2^36 -> -0x10 (FFFFFFF0) 00000000.
      });
    }
    assert(r == -0x10);
  }

  // mul_hi with negative result w/o carry
  {
    s::cl_int r{0};
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mul_hiSI1SI3>([=]() {
          AccR[0] = s::mul_hi(s::cl_int{-0x10000000}, s::cl_int{0x00000101});
        }); // -2^28 * (2^8 + 1) = -2^36 - 2^28 -> -0x11 (FFFFFFEF) -0x10000000
            // (F0000000).
      });
    }
    assert(r == -0x11);
  }

  // rotate
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class rotateSI1SI1>([=]() {
          AccR[0] = s::rotate(s::cl_int{ 0x11100000 }, s::cl_int{ 12 });
        });
      });
    }
    assert(r == 0x00000111);
  }

  // rotate (with large rotate size)
  {
    s::cl_char r{ 0 };
    {
      s::buffer<s::cl_char, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class rotateSI1SI2>([=]() {
          AccR[0] = s::rotate(static_cast<s::cl_char>((unsigned char)0xe0),
              s::cl_char{ 50 });
        });
      });
    }
    assert((unsigned char)r == 0x83);
  }
  // sub_sat
  {
    auto TestSubSat = [](s::cl_int x, s::cl_int y) {
      s::cl_int r{ 0 };
      {
        s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
        s::queue myQueue;
        myQueue.submit([&](s::handler &cgh) {
          auto AccR = BufR.get_access<s::access::mode::write>(cgh);
          cgh.single_task<class sub_satSI1SI1>([=]() {
            AccR[0] = s::sub_sat(x, y);
          });
        });
      }
      return r;
    };
    // 10 - (-2^31(minimum value)) = saturates on Maximum value
    s::cl_int r1 = TestSubSat(10, 0x80000000);
    assert(r1 == 0x7FFFFFFF);
    s::cl_int r2 = TestSubSat(0x7FFFFFFF, 0xFFFFFFFF);
    assert(r2 == 0x7FFFFFFF);
    s::cl_int r3 = TestSubSat(0x80000000, 0x00000001);
    assert(r3 == 0x80000000);
    s::cl_int r4 = TestSubSat(10499, 30678);
    assert(r4 == -20179);
  }

  // upsample - 1
  {
    s::cl_ushort r{ 0 };
    {
      s::buffer<s::cl_ushort, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleUC1UC1>([=]() {
          AccR[0] = s::upsample(s::cl_uchar{ 0x10 }, s::cl_uchar{ 0x10 });
        });
      });
    }
    assert(r == 0x1010);
  }

  // upsample - 2
  {
    s::cl_short r{ 0 };
    {
      s::buffer<s::cl_short, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleSC1UC1>([=]() {
          AccR[0] = s::upsample(s::cl_char{ 0x10 }, s::cl_uchar{ 0x10 });
        });
      });
    }
    assert(r == 0x1010);
  }

  // upsample - 3
  {
    s::cl_uint r{ 0 };
    {
      s::buffer<s::cl_uint, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleUS1US1>([=]() {
          AccR[0] = s::upsample(s::cl_ushort{ 0x0010 }, s::cl_ushort{ 0x0010 });
        });
      });
    }
    assert(r == 0x00100010);
  }

  // upsample - 4
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleSS1US1>([=]() {
          AccR[0] = s::upsample(s::cl_short{ 0x0010 }, s::cl_ushort{ 0x0010 });
        });
      });
    }
    assert(r == 0x00100010);
  }

  // upsample - 5
  {
    s::cl_ulong r{ 0 };
    {
      s::buffer<s::cl_ulong, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleUI1UI1>([=]() {
          AccR[0] =
              s::upsample(s::cl_uint{ 0x00000010 }, s::cl_uint{ 0x00000010 });
        });
      });
    }
    assert(r == 0x0000001000000010);
  }

  // upsample - 6
  {
    s::cl_long r{ 0 };
    {
      s::buffer<s::cl_long, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class upsampleSI1UI1>([=]() {
          AccR[0] =
              s::upsample(s::cl_int{ 0x00000010 }, s::cl_uint{ 0x00000010 });
        });
      });
    }
    assert(r == 0x0000001000000010);
  }

  // popcount
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class popcountSI1>([=]() {
          AccR[0] = s::popcount(s::cl_int{ 0x000000FF });
        });
      });
    }
    assert(r == 8);
  }

  // mad24
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mad24SI1SI1SI1>([=]() {
          AccR[0] =
              s::mad24(s::cl_int(0xFFFFFFFF), s::cl_int{ 20 }, s::cl_int{ 20 });
        });
      });
    }
    assert(r == 0);
  }

  // mul24
  {
    s::cl_int r{ 0 };
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class mul24SI1SI1>([=]() {
          AccR[0] = s::mul24(s::cl_int(0xFFFFFFFF), s::cl_int{ 20 });
        });
      });
    }
    assert(r == -20);
  }

  return 0;
}
