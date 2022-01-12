// RUN: not %clangxx -fsycl -fsycl-device-only -S %s -o %t 2>&1 | FileCheck %s

#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

// CHECK: error: function 'cl::sycl::multi_ptr<{{.+}}> cl::sycl::accessor<{{.+}}>::get_pointer<{{.+}}>() const' is not supported in ESIMD context

SYCL_EXTERNAL auto
test(accessor<int, 1, access::mode::read_write, access::target::device> &acc)
    SYCL_ESIMD_FUNCTION {
  return acc.get_pointer();
}
