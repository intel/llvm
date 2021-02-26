// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -o %t.out %s
// RUN: %RUN_ON_HOST %t.out

#include <CL/sycl.hpp>
#include <cassert>

constexpr auto sycl_read_write = cl::sycl::access::mode::read_write;
constexpr auto sycl_global_buffer = cl::sycl::access::target::global_buffer;

struct SecondBase {
  SecondBase(int _E) : E(_E) {}
  int E;
};

struct InnerFieldBase {
  InnerFieldBase(int _D) : D(_D) {}
  int D;
};

struct InnerField : public InnerFieldBase {
  InnerField(int _C, int _D) : C(_C), InnerFieldBase(_D) {}
  int C;
};

struct Base {
  Base(int _B, int _C, int _D) : B(_B), InnerObj(_C, _D) {}
  int B;
  InnerField InnerObj;
};

struct Derived : public Base, public SecondBase {
  Derived(
      int _A, int _B, int _C, int _D, int _E,
      cl::sycl::accessor<int, 1, sycl_read_write, sycl_global_buffer> &_Acc)
      : A(_A), Acc(_Acc), /*Out(_Out),*/ Base(_B, _C, _D), SecondBase(_E) {}
  void operator()() const {
    Acc[0] = this->A + this->B + this->InnerObj.C + this->InnerObj.D + this->E;
  }

  int A;
  cl::sycl::accessor<int, 1, sycl_read_write, sycl_global_buffer> Acc;
};

int main() {
  int A[] = {10};
  {
    cl::sycl::queue Q;
    cl::sycl::buffer<int, 1> Buf(A, 1);

    Q.submit([&](cl::sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      Derived F = {1, 2, 3, 4, 5, Acc /*, Out*/};
      cgh.single_task(F);
    });
  }
  assert(A[0] == 15);
  return 0;
}
