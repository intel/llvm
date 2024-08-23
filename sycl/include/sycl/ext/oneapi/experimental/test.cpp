#include <sycl/sycl.hpp>

using namespace sycl;

template
struct ptr

int main() {
    queue q;
    int magic;
    buffer<int, 1> b{&magic, 1};
    q.submit([&](handler& h) {
    sycl::accessor acc{b, h};
    local_accessor<int[][2], 0> acc{h};
    h.parallel_for(nd_range<2>({2, 2}, {1, 1}), [=](nd_item<2> it) {
        acc[it.get_local_id(0)][it.get_local_id(1)] = 42;
});
});
}
