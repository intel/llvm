// RUN: not %clangxx -fsycl -fsycl-device-only -flegacy-pass-manager -S %s -o /dev/null 2>&1 | FileCheck %s
// RUN: not %clangxx -fsycl -fsycl-device-only -fno-legacy-pass-manager -S %s -o /dev/null 2>&1 | FileCheck %s
// RUN: not %clangxx -fsycl -fsycl-device-only -flegacy-pass-manager -O0 -S %s -o /dev/null 2>&1 | FileCheck %s
// RUN: not %clangxx -fsycl -fsycl-device-only -fno-legacy-pass-manager -O0 -S %s -o /dev/null 2>&1 | FileCheck %s

#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

// CHECK-DAG: error: function 'cl::sycl::multi_ptr<{{.+}}> cl::sycl::accessor<{{.+}}>::get_pointer<{{.+}}>() const' is not supported in ESIMD context
// CHECK-DAG: error: function '{{.+}} cl::sycl::accessor<{{.+}}>::operator[]<{{.+}}>({{.+}}) const' is not supported in ESIMD context

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
