// Cross-platform dynamic library loading helpers.
// Included by test .inc files that need dlopen/LoadLibrary abstraction.

#ifdef _WIN32
#include <windows.h>

using lib_handle_t = HMODULE;

static lib_handle_t loadLib(const char *Path) {
  HMODULE h = LoadLibraryExA(Path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!h)
    std::fprintf(stderr, "LoadLibraryExA(%s) failed: error %lu\n", Path,
                 GetLastError());
  return h;
}

static void *getSymbol(lib_handle_t H, const char *Name) {
  return (void *)GetProcAddress(H, Name);
}

static void unloadLib(lib_handle_t H) { FreeLibrary(H); }

#else
#include <dlfcn.h>

using lib_handle_t = void *;

static lib_handle_t loadLib(const char *Path) {
  void *h = dlopen(Path, DLOPEN_FLAGS);
  if (!h)
    std::fprintf(stderr, "dlopen(%s) failed: %s\n", Path, dlerror());
  return h;
}

static void *getSymbol(lib_handle_t H, const char *Name) {
  return dlsym(H, Name);
}

static void unloadLib(lib_handle_t H) { dlclose(H); }
#endif
