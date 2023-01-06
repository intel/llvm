// RUN: %clang_cc1 -fsycl-is-device -sycl-std=2020 -verify %s

// This test verifies that an error is thrown if std::atomic 
// is used inside device code.

#include "Inputs/sycl.hpp"

namespace std {   
 const bool test = true;    
template< class T >   
struct atomic {       
  atomic() {}         
  void use() const {} 
  T t;                
};                    
} // namespace std   



using namespace sycl;
queue q;

int main() {                          
   const std::atomic<bool> flag;       
   q.submit([&](handler &h) {          
     h.single_task<class KernelA>([=] {
       flag.use();                     
       //const bool x = std::test;
     });                               
   });  
}                               
