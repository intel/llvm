// RUN: %clangxx -fsycl-device-only -fsycl-unnamed-lambda -emit-llvm %s -o %t1.bc
// RUN: llvm-dis %t1.bc -o - | FileCheck %s
// RUN: %clangxx -fsycl-device-only -fsycl-unnamed-lambda -emit-llvm %s -DUSE_WRAPPER=1 -o %t2.bc
// RUN: llvm-dis %t2.bc -o - | FileCheck %s

// Mangling of kernel lambda must be the same for both versions of half
// CHECK: __unique_stable_name{{.*}} = private unnamed_addr constant [52 x i8] c"_ZTSN2cl4sycl6bufferINS0_4pairIDF16_NS0_5dummyEEEEE\00"

// Helper function to get string returned by __unique_stable_name in LLVM IR
template <typename T>
void print() {
  auto temp = __unique_stable_name(T);
}

// Helper function to get "print" emitted in device code
template<typename T, typename F>
__attribute__((sycl_kernel)) void helper(F f) {
  print<T>();
  f();
}

// Half wrapper, as it defined in SYCL headers
namespace cl {
namespace sycl {
namespace detail {
namespace half_impl {
class half {
public:
  half operator=(int) {return *this;}
};
} // namespace half_impl
} // namespace detail
} // namespace sycl
} // namespace cl

#ifndef USE_WRAPPER
using half = _Float16;
#else
using half = cl::sycl::detail::half_impl::half;
#endif

// A few more fake data types to complicate the mangling
namespace cl {
namespace sycl {
struct dummy {
  int a;
};
template<typename T1, typename T2>
struct pair {
  T1 a;
  T2 b;
};
template <typename T>
class buffer {
public:
  T &operator[](int) const { return value; }
  mutable T value;
};
} // namespace sycl
} // namespace cl

int main() {
  cl::sycl::buffer<cl::sycl::pair<half, cl::sycl::dummy>> B1;

  helper<decltype(B1)>([](){});

  return 0;
}
