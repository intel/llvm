// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fsycl-std-layout-kernel-params -Wno-sycl-2017-compat -fsyntax-only -fsycl-int-footer=%t.h %s
// RUN: FileCheck -input-file=%t.h %s
//
// This test checks if compiler reports compilation error on an attempt to pass
// non-standard layout struct object as SYCL kernel parameter.

struct Base {
  int X;
};

// This struct has non-standard layout, because both C (the most derived class)
// and Base have non-static data members.
struct C : public Base {
  int Y;
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

struct Kernel {
  void operator()() const {
    (void) c1;
    (void) c2;
    (void) p;
    (void) q;
  }

  int p;
  C c1;

  int q;
  C c2;
};

void test_struct_field() {
  Kernel k{};

  kernel_single_task<class kernel_object>(k);
}

// CHECK-LABEL: #include <CL/sycl/detail/sycl_fe_intrins.hpp>
// CHECK: static_assert(::sycl::is_device_copyable<C>_v, "error: kernel parameter type ('C') is not device copyable");
