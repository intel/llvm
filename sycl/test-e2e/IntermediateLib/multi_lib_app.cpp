// UNSUPPORTED: cuda || hip
// UNSUPPORTED-TRACKER: CMPLRLLVM-69415

// DEFINE: %{fPIC_flag} =  %if windows %{%} %else %{-fPIC%}
// DEFINE: %{shared_lib_ext} = %if windows %{dll%} %else %{so%}

// clang-format off
// IMPORTANT   -DSO_PATH='R"(%t.dir)"'
//              We need to capture %t.dir, the build directory, in a string
//              and the normal STRINGIFY() macros hack won't work.
//              Because on Windows, the path delimiters are \, 
//              which C++ preprocessor converts to escape sequences, 
//              which becomes a nightmare.
//              So the hack here is to put heredoc in the definition
//              and use single quotes, which Python forgivingly accepts.  
// clang-format on 

// On Windows, the CI sometimes builds on one machine and runs on another.
// This means that %t.dir might not be consistent between build and run.
// So we use %{run-aux} to perform ALL actions on the run machine 
// like we do for the AoT tests.

// RUN: rm -rf %t.dir ; mkdir -p %t.dir 
// RUN: %{run-aux} %clangxx -fsycl  %{fPIC_flag} -DSO_PATH='R"(%t.dir)"' -o %t.out %s

// RUN:  %{run-aux} %clangxx -fsycl %{fPIC_flag} -shared -DINC=1 -o %t.dir/lib_a.%{shared_lib_ext} %S/Inputs/incrementing_lib.cpp
// RUN:  %{run-aux} %clangxx -fsycl %{fPIC_flag} -shared -DINC=2 -o %t.dir/lib_b.%{shared_lib_ext} %S/Inputs/incrementing_lib.cpp
// RUN:  %{run-aux} %clangxx -fsycl %{fPIC_flag} -shared -DINC=4 -o %t.dir/lib_c.%{shared_lib_ext} %S/Inputs/incrementing_lib.cpp

// RUN:  env UR_L0_LEAKS_DEBUG=1 %{run} %t.out

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

// This test uses a kernel of the same name in three different shared libraries.
// It loads each library, calls the kernel, and checks that the incrementation
// is done correctly, and then unloads the library.
// It also reloads the first library after unloading it. 
// This test ensures that __sycl_register_lib() and __sycl_unregister_lib()
// are called correctly, and that the device images are cleaned up properly.


#include <sycl/detail/core.hpp>

using namespace sycl::ext::oneapi::experimental;


#ifdef _WIN32
#include <windows.h>

void *loadOsLibrary(const std::string &LibraryPath) {
  HMODULE h =
      LoadLibraryExA(LibraryPath.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!h) {
    std::cout << "LoadLibraryExA(" << LibraryPath
              << ") failed with error code " << GetLastError() << std::endl;
  }
  return (void *)h;
}
int unloadOsLibrary(void *Library) {
  return FreeLibrary((HMODULE)Library) ? 0 : 1;
}
void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName) {
  return (void *)GetProcAddress((HMODULE)Library, FunctionName.c_str());
}

#else
#include <dlfcn.h>

void *loadOsLibrary(const std::string &LibraryPath) {
  void *so = dlopen(LibraryPath.c_str(), RTLD_NOW);
  if (!so) {
    char *Error = dlerror();
    std::cerr << "dlopen(" << LibraryPath << ") failed with <"
              << (Error ? Error : "unknown error") << ">" << std::endl;
  }
  return so;
}

int unloadOsLibrary(void *Library) { return dlclose(Library); }

void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName) {
  return dlsym(Library, FunctionName.c_str());
}
#endif

// Define the function pointer type for performIncrementation
using IncFuncT = void(sycl::queue &, sycl::buffer<int, 1> &);

void initializeBuffer(sycl::buffer<int, 1> &buf) {
  auto acc = sycl::host_accessor<int, 1>(buf);
  for (size_t i = 0; i < buf.size(); ++i)
    acc[i] = 0;
}

void checkIncrementation(sycl::buffer<int, 1> &buf, int val) {
  auto acc = sycl::host_accessor<int, 1>(buf);
  for (size_t i = 0; i < buf.size(); ++i) {
    std::cout << acc[i] << " ";
    assert(acc[i] == val);
  }
  std::cout << std::endl;
}

int main() {
  sycl::queue q;

  sycl::range<1> r(8);
  sycl::buffer<int, 1> buf(r);
  initializeBuffer(buf);

  std::string base_path = SO_PATH;

#ifdef _WIN32
  std::string path_to_lib_a = base_path + "\\lib_a.dll";
  std::string path_to_lib_b = base_path + "\\lib_b.dll";
  std::string path_to_lib_c = base_path + "\\lib_c.dll";
#else
  std::string path_to_lib_a = base_path + "/lib_a.so";
  std::string path_to_lib_b = base_path + "/lib_b.so";
  std::string path_to_lib_c = base_path + "/lib_c.so";
#endif

  std::cout << "paths: " << path_to_lib_a << std::endl;
  std::cout << "SO_PATH: " << SO_PATH << std::endl;

  void *lib_a = loadOsLibrary(path_to_lib_a);
  void *f = getOsLibraryFuncAddress(lib_a, "performIncrementation");
  if(!f){ 
    std::cout << "Cannot get performIncremenation function from .so/.dll" << std::endl;
    return 1;
  }
  auto performIncrementationFuncA = reinterpret_cast<IncFuncT *>(f);
  performIncrementationFuncA(q, buf); // call the function from lib_a
  q.wait();
  checkIncrementation(buf, 1);
  unloadOsLibrary(lib_a);
  std::cout << "lib_a done" << std::endl;


  // Now RELOAD lib_a and try it again.
  lib_a = loadOsLibrary(path_to_lib_a);
  f = getOsLibraryFuncAddress(lib_a, "performIncrementation");
  performIncrementationFuncA = reinterpret_cast<IncFuncT *>(f);
  performIncrementationFuncA(q, buf); // call the function from lib_a
  q.wait();
  checkIncrementation(buf, 1 + 1);
  unloadOsLibrary(lib_a);
  std::cout << "reload of lib_a done" << std::endl;


  void *lib_b = loadOsLibrary(path_to_lib_b);
  f = getOsLibraryFuncAddress(lib_b, "performIncrementation");
  auto performIncrementationFuncB = reinterpret_cast<IncFuncT *>(f);
  performIncrementationFuncB(q, buf); // call the function from lib_b
  q.wait();
  checkIncrementation(buf, 1 + 1 + 2);
  unloadOsLibrary(lib_b);
  std::cout << "lib_b done" << std::endl;

  void *lib_c = loadOsLibrary(path_to_lib_c);
  f = getOsLibraryFuncAddress(lib_c, "performIncrementation");
  auto performIncrementationFuncC = reinterpret_cast<IncFuncT *>(f);
  q.wait();
  performIncrementationFuncC(q, buf); // call the function from lib_c
  checkIncrementation(buf, 1 + 1 + 2 + 4);
  unloadOsLibrary(lib_c);
  std::cout << "lib_c done" << std::endl;

  return 0;
}
