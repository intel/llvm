#include <CL/sycl.hpp>

#include <cassert>

using namespace cl::sycl;

template <typename T, typename convertT, int roundingMode>
class kernel_name;

template <int N>
struct helper;

template <>
struct helper<0> {
  template <typename T, int NumElements>
  static void compare(const vec<T, NumElements> &x,
                      const vec<T, NumElements> &y) {
    const T xs = x.template swizzle<0>();
    const T ys = y.template swizzle<0>();
    assert(xs == ys);
  }
};

template <int N>
struct helper {
  template <typename T, int NumElements>
  static void compare(const vec<T, NumElements> &x,
                      const vec<T, NumElements> &y) {
    const T xs = x.template swizzle<N>();
    const T ys = y.template swizzle<N>();
    helper<N - 1>::compare(x, y);
    assert(xs == ys);
  }
};

template <typename T, typename convertT, int NumElements,
          rounding_mode roundingMode>
void test(const vec<T, NumElements> &ToConvert,
          const vec<convertT, NumElements> &Expected) {
  vec<convertT, NumElements> Converted{0};
  {
    buffer<vec<convertT, NumElements>, 1> Buffer{&Converted, range<1>{1}};
    queue Queue;
    Queue.submit([&](handler &CGH) {
      accessor<vec<convertT, NumElements>, 1, access::mode::write> Accessor(
          Buffer, CGH);
        CGH.single_task<class kernel_name<T, convertT, static_cast<int>(roundingMode)>>([=]() {
          Accessor[0] = ToConvert.template convert<convertT, roundingMode>();
        });
    });
  }
  helper<NumElements - 1>::compare(Converted, Expected);
}
