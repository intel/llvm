// RUN: %clang_cc1 -fsycl-is-device -sycl-std=2020 -verify %s

// This test verifies that an error is thrown if std::atomic 
// is used inside device code.

#include "Inputs/sycl.hpp"

namespace std {
struct int8_t;
template< class T >   
struct atomic {       
  atomic() {}         
};                    
} // namespace std   

using namespace sycl;
queue q;

void usage() {

    // expected-error@+1 {{std::atomic type is not supported in device code}}
    std::atomic<char> AtomicChar;
    // expected-error@+1 {{std::atomic type is not supported in device code}}
    std::atomic<bool> AtomicBool;
    // expected-error@+1 {{std::atomic type is not supported in device code}}
    std::atomic<std::int8_t> AtomicInt8_t;
}

int main() {                         
   // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<KernelA, (lambda}}
   q.submit([&](handler &h) {         
     h.single_task<class KernelA>([=] {
      // expected-note@+1 {{called by 'operator()'}}
        usage();
     });                               
   });  
}                               
