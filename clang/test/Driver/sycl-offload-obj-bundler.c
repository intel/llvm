///
/// Generate host and target object and bundle them into a single object file.
///

// REQUIRES: clang-driver
// REQUIRES: x86-registered-target

/// ###########################################################################

// RUN:   %clang -c -fsycl %s 

/// ###########################################################################
int j = 9;
int foo() {

   return j;
}
