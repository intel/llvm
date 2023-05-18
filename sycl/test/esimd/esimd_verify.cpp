// RUN: not %clangxx -fsycl -fsycl-device-only -S %s -o /dev/null 2>&1 | FileCheck %s
// RUN: not %clangxx -fsycl -fsycl-device-only -O0 -S %s -o /dev/null 2>&1 | FileCheck %s

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

// CHECK-DAG: error: function 'int* sycl::_V1::accessor<{{.+}}>::get_pointer<{{.+}}>() const' is not supported in ESIMD context
// CHECK-DAG: error: function '{{.+}} sycl::_V1::accessor<{{.+}}>::operator[]<{{.+}}>({{.+}}) const' is not supported in ESIMD context
// CHECK-DAG: error: function '{{.+}}combine(int const&)' is not supported in ESIMD context

SYCL_EXTERNAL auto
test(accessor<int, 1, access::mode::read_write, access::target::device> &acc)
    SYCL_ESIMD_FUNCTION {
  return acc.get_pointer();
}

SYCL_EXTERNAL void
test1(accessor<int, 1, access::mode::read_write, access::target::device> &acc)
    SYCL_ESIMD_FUNCTION {
  acc[0] = 0;
}

void test2(sycl::handler &cgh, int *buf) {
  auto reduction = sycl::reduction(buf, sycl::plus<int>());
  cgh.parallel_for<class Test2>(sycl::range<1>(1), reduction,
                                [=](sycl::id<1>, auto &reducer)
                                    SYCL_ESIMD_KERNEL { reducer.combine(15); });
}
