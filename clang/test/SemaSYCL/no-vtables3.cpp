// RUN: %clang_cc1 -fsycl-is-device -Wno-return-type -verify -fsyntax-only -x c++ -emit-llvm-only %s

struct Base {
  virtual void f() const {}
};

// expected-error@+1 3{{SYCL kernel cannot have a class with a virtual function table}}
struct Inherit : Base {
  virtual void f() const override {}
};

struct Wrapper{
  Wrapper() {
    // expected-note@+1{{used here}}
    Inherit IH;
  }

  void Func() {
    // expected-note@+1{{used here}}
    Inherit IH;
  }

  ~Wrapper() {
  // expected-note@+1{{used here}}
    Inherit IH;
  }
};

void usage() {
  Wrapper WR;
  WR.Func();
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}
int main() {
  kernel_single_task<class fake_kernel>([]() { usage(); });
  return 0;
}

