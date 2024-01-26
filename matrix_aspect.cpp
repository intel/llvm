#include <sycl/sycl.hpp>

using namespace sycl;
int main() {
   queue q;
   auto d = q.get_device();
   std::cout << std::boolalpha << d.has(aspect::ext_intel_matrix);
   return 0;
}
