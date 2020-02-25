// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %clangxx -fsycl -D HALF_IS_SUPPORTED %s -o %t_gpu.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t_gpu.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// XFAIL: cuda
#include <CL/sycl.hpp>

#include <cassert>

namespace s = cl::sycl;
using namespace std;

template <typename T, typename R, bool Expected = true> void test_nan_call() {
  static_assert(is_same<decltype(s::nan(T{0})), R>::value == Expected, "");
}

template <typename, typename> struct test;

template <typename T, typename R> void check_nan(s::queue &Queue) {
  R Data{0};
  s::vec<R, 2> VData{0};
  {
    s::buffer<R, 1> Buf(&Data, s::range<1>(1));
    s::buffer<s::vec<R, 2>, 1> VBuf(&VData, s::range<1>(1));
    Queue.submit([&](s::handler &Cgh) {
      auto Acc = Buf.template get_access<s::access::mode::write>(Cgh);
      auto VAcc = VBuf.template get_access<s::access::mode::write>(Cgh);
      Cgh.single_task<test<T, R>>([=]() {
        Acc[0] = s::nan(T{0});
        VAcc[0] = s::nan(s::vec<T, 2>{0});
      });
    });
    Queue.wait_and_throw();
  }
  assert(s::isnan(Data));
  assert(s::all(s::isnan(VData)));
}

int main() {
  test_nan_call<s::ushort, half>();
  test_nan_call<s::uint, float>();
  test_nan_call<s::ulong, double>();
  test_nan_call<s::ulonglong, double>();
  test_nan_call<s::ushort2, s::half2>();
  test_nan_call<s::uint2, s::float2>();
  test_nan_call<s::ulong2, s::double2>();
  test_nan_call<s::ulonglong2, s::double2>();

  s::queue Queue([](cl::sycl::exception_list ExceptionList) {
    for (cl::sycl::exception_ptr_class ExceptionPtr : ExceptionList) {
      try {
        std::rethrow_exception(ExceptionPtr);
      } catch (cl::sycl::exception &E) {
        std::cerr << E.what() << std::endl;
      } catch (...) {
        std::cerr << "Unknown async exception was caught." << std::endl;
      }
    }
  });
#ifdef HALF_IS_SUPPORTED
  if (Queue.get_device().has_extension("cl_khr_fp16"))
    check_nan<unsigned short, half>(Queue);
#endif
  check_nan<unsigned int, float>(Queue);
  if (Queue.get_device().has_extension("cl_khr_fp64")) {
    check_nan<unsigned long, double>(Queue);
    check_nan<unsigned long long, double>(Queue);
  }
  return 0;
}
