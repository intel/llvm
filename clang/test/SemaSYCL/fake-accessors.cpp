// RUN: %clang_cc1 -fsycl-is-device -Wno-int-to-void-pointer-cast -internal-isystem %S/Inputs -ast-dump -sycl-std=2020 %s | FileCheck %s

#include "sycl.hpp"

sycl::queue deviceQueue;

namespace fake {
namespace cl {
namespace sycl {
class accessor {
public:
  int field;
};
} // namespace sycl
} // namespace cl
} // namespace fake

class accessor {
public:
  int field;
};

int main() {

  fake::cl::sycl::accessor FakeAccessor = {1};
  accessor AccessorClass = {1};

  typedef sycl::accessor<int, 1, sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>
      MyAccessorTD;
  MyAccessorTD AccessorTypeDef;

  using MyAccessorA = sycl::accessor<int, 1, sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>;
  MyAccessorA AccessorAlias;

  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> AccessorRegular;

  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class fake_accessors>(
        [=] {
          AccessorRegular.use((void *)(FakeAccessor.field + AccessorClass.field));
        });

    h.single_task<class accessor_typedef>(
        [=] {
          AccessorTypeDef.use((void *)(FakeAccessor.field + AccessorClass.field));
        });

    h.single_task<class accessor_alias>(
        [=] {
          AccessorAlias.use((void *)(FakeAccessor.field + AccessorClass.field));
        });
  });

  return 0;
}
// CHECK: fake_accessors{{.*}} 'void (__global int *, sycl::range<1>, sycl::range<1>, sycl::id<1>, fake::cl::sycl::accessor, accessor)
// CHECK: accessor_typedef{{.*}} 'void (__global int *, sycl::range<1>, sycl::range<1>, sycl::id<1>, fake::cl::sycl::accessor, accessor)
// CHECK: accessor_alias{{.*}} 'void (__global int *, sycl::range<1>, sycl::range<1>, sycl::id<1>, fake::cl::sycl::accessor, accessor)
