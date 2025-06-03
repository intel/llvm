// RUN: %clangxx -fsyntax-only -fsycl %s

// Verify that we can pass top-level special type parameters to free function
// kernels.

#include <sycl/sycl.hpp>

struct Baz {
//  int a;
  sycl::accessor<int, 1> accA;
  sycl::accessor<int, 1> accB;
  sycl::accessor<int, 1> accC;
};

struct Foo {
  Baz b;
  sycl::accessor<int, 1> Acc;
};

Baz b;

using namespace sycl;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void foo(Baz b) {
  for (int i = 0; i < 10; ++i) {
    b.accC[i] = b.accA[i] * b.accB[i] + 25;
  }
    //str << "Done!" << sycl::endl;
} /*, sampler S,
         stream str, ext::oneapi::experimental::annotated_arg<int> arg,
         ext::oneapi::experimental::annotated_ptr<int> ptr, Foo f) {}*/

int main() {
  int Data[10];
  for (int i = 0; i < 10; ++i) {
    Data[i] = i;
  }
  int Sum[10];
  {
    buffer<int> bufA(&Data[0], 10);
    buffer<int> bufB(&Data[0], 10);
    buffer<int> bufC(&Sum[0], 10);
    queue Q;
    kernel_bundle bundle =
        get_kernel_bundle<bundle_state::executable>(Q.get_context());
    kernel_id id = ext::oneapi::experimental::get_kernel_id<foo>();
    kernel Kernel = bundle.get_kernel(id);
    Q.submit([&](handler &h) {
      sycl::stream str(8192, 1024, h);
      accessor<int, 1> accA(bufA, h);
      accessor<int, 1> accB(bufB, h);
      accessor<int, 1> accC(bufC, h);
      // local_accessor<int, 1> lacc;
      // local_accessor<int, 1> macc;
      b = Baz{accA, accB, accC};
      // Foo f;
      h.set_args(b);
      h.parallel_for(nd_range{{1}, {1}}, Kernel);
    });
  }
for (int i = 0;i < 10; ++i) {
    std::cout << Sum[i] << std::endl;
}
  return 0;
}
