// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsyntax-only -verify %s
// expected-no-diagnostics

#include <sycl.hpp>

using namespace cl::sycl;

namespace s = sycl;

struct P {
    union S {
        int32_t p;
    } q;
};

static_assert(std::is_trivially_copyable<P>::value, "");
static_assert(std::is_trivially_copyable<decltype(P{}.q)>::value, "");

int main() {
    s::queue q;

    P p = {};

    q.submit([&](s::handler& h) {
        h.single_task<class foo>([=](){
            (void) p;
        });
    });
}

