// DEFINE: %{fPIC_flag} =  %if windows %{%} %else %{-fPIC%}
// DEFINE: %{shared_lib_ext} = %if windows %{dll%} %else %{so%}

// RUN: %{build} %{fPIC_flag} -DSO_PATH="%T/" -o %t.out
// RUN:  %clangxx -fsycl %{fPIC_flag} -shared -DINC=1 -o %T/lib_a.%{shared_lib_ext} %S/Inputs/incrementing_lib.cpp
// RUN:  %clangxx -fsycl %{fPIC_flag} -shared -DINC=2 -o %T/lib_b.%{shared_lib_ext} %S/Inputs/incrementing_lib.cpp
// RUN:  %clangxx -fsycl %{fPIC_flag} -shared -DINC=4 -o %T/lib_c.%{shared_lib_ext} %S/Inputs/incrementing_lib.cpp

// RUN:  env UR_L0_LEAKS_DEBUG=1 %{run} %t.out

// This test uses a kernel of the same name in three different shared libraries.
// It loads each library, calls the kernel, and checks that the incrementation
// is done correctly, and then unloads the library.
// This test ensures that __sycl_register_lib() and __sycl_unregister_lib()
// are called correctly, and that the device images are cleaned up properly.

#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

#define STRINGIFY_HELPER(A) #A
#define STRINGIFY(A) STRINGIFY_HELPER(A)
#define SO_FNAME "" STRINGIFY(SO_PATH) ""

#ifdef _WIN32
#include <windows.h>

void *loadOsLibrary(const std::string &LibraryPath) {
  HMODULE h =
      LoadLibraryExA(LibraryPath.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
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

  std::string base_path = SO_FNAME;

#ifdef _WIN32
  std::string path_to_lib_a = base_path + "lib_a.dll";
  std::string path_to_lib_b = base_path + "lib_b.dll";
  std::string path_to_lib_c = base_path + "lib_c.dll";
#else
  std::string path_to_lib_a = base_path + "lib_a.so";
  std::string path_to_lib_b = base_path + "lib_b.so";
  std::string path_to_lib_c = base_path + "lib_c.so";
#endif

  void *lib_a = loadOsLibrary(path_to_lib_a);
  void *f = getOsLibraryFuncAddress(lib_a, "performIncrementation");
  auto performIncrementationFuncA = reinterpret_cast<IncFuncT *>(f);
  performIncrementationFuncA(q, buf); // call the function from lib_a
  q.wait();
  checkIncrementation(buf, 1);
  unloadOsLibrary(lib_a);
  std::cout << "lib_a done" << std::endl;

  void *lib_b = loadOsLibrary(path_to_lib_b);
  f = getOsLibraryFuncAddress(lib_b, "performIncrementation");
  auto performIncrementationFuncB = reinterpret_cast<IncFuncT *>(f);
  performIncrementationFuncB(q, buf); // call the function from lib_b
  q.wait();
  checkIncrementation(buf, 1 + 2);
  unloadOsLibrary(lib_b);
  std::cout << "lib_b done" << std::endl;

  void *lib_c = loadOsLibrary(path_to_lib_c);
  f = getOsLibraryFuncAddress(lib_c, "performIncrementation");
  auto performIncrementationFuncC = reinterpret_cast<IncFuncT *>(f);
  q.wait();
  performIncrementationFuncC(q, buf); // call the function from lib_c
  checkIncrementation(buf, 1 + 2 + 4);
  unloadOsLibrary(lib_c);
  std::cout << "lib_c done" << std::endl;

  return 0;
}