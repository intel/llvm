// RUN: %clang_cc1 -fsycl-is-device -sycl-std=2020 -verify -pedantic -fsyntax-only %s
// RUN: %clang_cc1 -verify -pedantic -fsyntax-only %s

class A {
public:
  A(int a) : m_val(a) {};
  A(const A &a) { m_val = a.m_val; }
private:
    int m_val;
};

struct st {
  int a;
  float b;
};


#ifdef __SYCL_DEVICE_ONLY__
static_assert(__has_builtin(__builtin_intel_fpga_reg), "");

struct inner {
  void (*fp)(); // expected-note {{field with illegal type declared here}}
};

struct outer {
  inner A;
};

void foo() {
  int a = 123;
  int b = __builtin_intel_fpga_reg(a);
  int c = __builtin_intel_fpga_reg(2.0f);
  int d = __builtin_intel_fpga_reg( __builtin_intel_fpga_reg( b+12 ));
  int e = __builtin_intel_fpga_reg( __builtin_intel_fpga_reg( a+b ));
  float f = 3.4f;
  int g = __builtin_intel_fpga_reg((int)f);
  A h(5);
  A j = __builtin_intel_fpga_reg(h);
  struct st i = {1, 5.0f};
  struct st ii = __builtin_intel_fpga_reg(i);
  int *ap = &a;
  int *bp = __builtin_intel_fpga_reg(ap);
  int intArr[10] = {0};
  int *k = __builtin_intel_fpga_reg(intArr);
  // expected-error@-1{{illegal argument of type 'int [10]'  to __builtin_intel_fpga_reg}}

  void (*fp1)();
  void (*fp2)() = __builtin_intel_fpga_reg(fp1);
  //expected-error@-1{{illegal argument of type 'void (*)()'  to __builtin_intel_fpga_reg}}
  struct outer iii;
  struct outer iv = __builtin_intel_fpga_reg(iii);
  //expected-error@-1{{illegal field in argument to __builtin_intel_fpga_reg}}
  void *vp = __builtin_intel_fpga_reg();
  // expected-error@-1{{too few arguments to function call, expected 1, have 0}}
  int tmp = __builtin_intel_fpga_reg(1, 2);
  // expected-error@-1{{too many arguments to function call, expected 1, have 2}}
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}
int main() {
  kernel_single_task<class fake_kernel>([]() { foo(); });
  return 0;
}

#else

static_assert(!__has_builtin(__builtin_intel_fpga_reg), "");
int main() {
  A a(3);
  A b = __builtin_intel_fpga_reg(a);
  // expected-error@-1{{'__builtin_intel_fpga_reg' is only available in SYCL device}}
  return 0;
}

#endif
