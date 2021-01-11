// RUN: %clang_cc1 -fsycl -fsycl-is-device -ast-dump -sycl-std=2020 %s | FileCheck %s

#include "Inputs/sycl.hpp"

sycl::queue myQueue;

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
  accessor acc1 = {1};

  sycl::accessor<int, 1, sycl::access::mode::read_write> accessorA;
  sycl::accessor<int, 1, sycl::access::mode::read_write> accessorB;
  sycl::accessor<int, 1, sycl::access::mode::read_write> accessorC;

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class fake_accessors>(
        [=] {
          accessorA.use((void *)(FakeAccessor.field + acc1.field));
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class accessor_typedef>(
        [=] {
          accessorB.use((void *)(FakeAccessor.field + acc1.field));
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class accessor_alias>(
        [=] {
          accessorC.use((void *)(FakeAccessor.field + acc1.field));
        });
  });

  return 0;
}
// CHECK: fake_accessors{{.*}} 'void (__global int *, sycl::range<1>, sycl::range<1>, sycl::id<1>, fake::cl::sycl::accessor, accessor)
// CHECK: accessor_typedef{{.*}} 'void (__global int *, sycl::range<1>, sycl::range<1>, sycl::id<1>, fake::cl::sycl::accessor, accessor)
// CHECK: accessor_alias{{.*}} 'void (__global int *, sycl::range<1>, sycl::range<1>, sycl::id<1>, fake::cl::sycl::accessor, accessor)
