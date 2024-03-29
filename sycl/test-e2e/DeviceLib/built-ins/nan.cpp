// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -fsycl-device-code-split=per_kernel -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}

#include <iostream>
#include <cassert>

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

namespace s = sycl;
using namespace std;

template <typename T, typename R, bool Expected = true> void test_nan_call() {
  static_assert(is_same<decltype(s::nan(T{0})), R>::value == Expected, "");
}

template <typename, typename> struct test_scalar;

template <typename T, typename R> void check_scalar_nan(s::queue &Queue) {
  R Data{0};
  {
    s::buffer<R, 1> Buf(&Data, s::range<1>(1));
    Queue.submit([&](s::handler &Cgh) {
      auto Acc = Buf.template get_access<s::access::mode::write>(Cgh);
      Cgh.single_task<test_scalar<T, R>>([=]() { Acc[0] = s::nan(T{0}); });
    });
    Queue.wait_and_throw();
  }
  assert(s::isnan(Data));
}

template <typename, typename> struct test_vec;

template <typename T, typename R> void check_vec_nan(s::queue &Queue) {
  s::vec<R, 2> VData{0};
  {
    s::buffer<s::vec<R, 2>, 1> VBuf(&VData, s::range<1>(1));
    Queue.submit([&](s::handler &Cgh) {
      auto VAcc = VBuf.template get_access<s::access::mode::write>(Cgh);
      Cgh.single_task<test_vec<T, R>>(
          [=]() { VAcc[0] = s::nan(s::vec<T, 2>{0}); });
    });
    Queue.wait_and_throw();
  }
  assert(s::all(s::isnan(VData)));
}

int main() {
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  test_nan_call<unsigned short, s::half>();
  test_nan_call<unsigned int, float>();
  test_nan_call<unsigned long, double>();
  test_nan_call<unsigned long long, double>();
  test_nan_call<s::ushort2, s::half2>();
  test_nan_call<s::uint2, s::float2>();
  test_nan_call<s::ulong2, s::double2>();
  test_nan_call<s::vec<unsigned long long, 2>, s::double2>();
#else
  test_nan_call<uint16_t, s::half>();
  test_nan_call<uint32_t, float>();
  test_nan_call<uint64_t, double>();
  test_nan_call<s::vec<uint16_t, 2>, s::half2>();
  test_nan_call<s::vec<uint32_t, 2>, s::float2>();
  test_nan_call<s::vec<uint64_t, 2>, s::double2>();
#endif

  s::queue Queue([](sycl::exception_list ExceptionList) {
    for (std::exception_ptr ExceptionPtr : ExceptionList) {
      try {
        std::rethrow_exception(ExceptionPtr);
      } catch (sycl::exception &E) {
        std::cerr << E.what() << std::endl;
      } catch (...) {
        std::cerr << "Unknown async exception was caught." << std::endl;
      }
    }
  });

  if (Queue.get_device().has(sycl::aspect::fp16)) {
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
    check_scalar_nan<unsigned short, s::half>(Queue);
    check_vec_nan<uint16_t, s::half>(Queue);
#else
    check_scalar_nan<uint16_t, s::half>(Queue);
    check_vec_nan<uint16_t, s::half>(Queue);
#endif
  }

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  check_scalar_nan<unsigned int, float>(Queue);
  check_vec_nan<uint32_t, float>(Queue);
#else
  check_scalar_nan<uint32_t, float>(Queue);
  check_vec_nan<uint32_t, float>(Queue);
#endif

  if (Queue.get_device().has(sycl::aspect::fp64)) {
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
    check_scalar_nan<unsigned long, double>(Queue);
    check_scalar_nan<unsigned long long, double>(Queue);
    check_vec_nan<uint64_t, double>(Queue);
    check_vec_nan<unsigned long long, double>(Queue);
#else
    check_scalar_nan<uint64_t, double>(Queue);
    check_vec_nan<uint64_t, double>(Queue);
#endif
  }
  return 0;
}
