// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -Wno-sycl-2017-compat -verify %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

class Class {
public:
  Class() : member(1) {}
  void function();
  int member;
};

void Class::function() {
  // expected-note@+1{{used here}}
  kernel<class kernel_wrapper>(
      [=]() {
        int acc[1] = {5};
        acc[0] *= member; // expected-error{{implicit capture of 'this' is not allowed for kernel functions}}
      });
}

int main(int argc, char *argv[]) {
  Class c;
  c.function();
}
