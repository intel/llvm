// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -c -o %t.o

#include <CL/sycl.hpp>

#include <type_traits>

// Ensure that <CL/sycl.hpp> provides the same functionality as <sycl/sycl.hpp>
// under "cl" namespace.

static_assert(std::is_same_v<cl::sycl::queue, sycl::queue>);
