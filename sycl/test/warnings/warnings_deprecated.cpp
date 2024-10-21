// RUN: %clangxx -fsycl -sycl-std=2020 -fsycl-device-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -fsyntax-only -Wall -Wdeprecated -Wextra

#include <sycl/sycl.hpp>


using namespace sycl;
queue q;


class FunctorDeprecated {
public:
  // expected-warning@+1{{attribute 'intel::reqd_sub_group_size' is deprecated}}
  [[intel::reqd_sub_group_size(16)]] void operator()() const {}
};

int main() {

  q.submit([&](handler &h) {
    FunctorDeprecated fDeprecated;
    h.single_task<class kernel_name1>(fDeprecated);
  });

  return 0;
}