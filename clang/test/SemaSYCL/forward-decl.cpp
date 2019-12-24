// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -pedantic %s

// expected-no-diagnostics
template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
      class Foo *F;
      class Boo {
      public:
        virtual int getBoo() { return 42; }
      };
  });

  kernel_single_task<class kernel_function_2>([]() {
      class Boo *B;
  });
  return 0;
}
