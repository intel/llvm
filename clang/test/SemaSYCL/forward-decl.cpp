// RUN: %clang_cc1 -x c++ -DNOVIRTUAL -fsycl-is-device -std=c++11 -fsyntax-only -verify -pedantic %s
// RUN: %clang_cc1 -x c++ -DVIRTUAL -fsycl-is-device -std=c++11 -fsyntax-only -verify -pedantic %s

// expected-no-diagnostics
template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
#ifdef NOVIRTUAL
  kernel_single_task<class kernel_function_1>([]() {
      class Foo *F;
  });
#elif VIRTUAL
  kernel_single_task<class kernel_function_2>([]() {
      class Boo {
      public:
        virtual int getBoo() { return 42; }
      };
  });

  kernel_single_task<class kernel_function_3>([]() {
      class Boo *B;
  });
#endif // VIRTUAL
  return 0;
}
