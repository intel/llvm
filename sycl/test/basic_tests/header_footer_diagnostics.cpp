// RUN: %clangxx -fsycl -Werror=reserved-identifier -Werror=old-style-cast %s

// Check that the generated header and footer files do not generate
// errors when pedantic warnings are enabled.

#include <sycl/sycl.hpp>
sycl::ext::oneapi::experimental::device_global<int> devGlobVar;
int main(int argc, char **argv) { return 0; }
