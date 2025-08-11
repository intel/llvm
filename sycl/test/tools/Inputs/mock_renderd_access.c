#define _GNU_SOURCE

#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <glob.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

const char *renderd = "/dev/dri/renderD*";
const char *mock_render_dirs[2] = {"/dev/dri/renderD128",
                                   "/dev/dri/renderD129"};

int glob(const char *pattern, int flags, int (*errfunc)(const char *, int),
         glob_t *pglob) {
  const char *mock_mode = getenv("MOCK_GLOB_MODE");
  if (!mock_mode || (strcmp(pattern, renderd) != 0)) {
    // Delegate to real glob
    int (*real_glob)(const char *, int, int (*)(const char *, int), glob_t *);
    real_glob = dlsym(RTLD_NEXT, "glob");
    if (!real_glob) {
      errno = ENOSYS;
      return -1;
    }
    return real_glob(pattern, flags, errfunc, pglob);
  }
  if (strcmp(mock_mode, "notfound") == 0) {
    pglob->gl_pathc = 0;
    pglob->gl_pathv = NULL;
    return 0;
  }
  assert(strcmp(mock_mode, "exists") == 0);
  pglob->gl_pathc = 2;
  pglob->gl_pathv = mock_render_dirs;
  return 0;
}

int glob64(const char *pattern, int flags, int (*errfunc)(const char *, int),
           glob64_t *pglob) {
  const char *mock_mode = getenv("MOCK_GLOB_MODE");
  if (!mock_mode || (strcmp(pattern, renderd) != 0)) {
    // Delegate to real glob64
    int (*real_glob64)(const char *, int, int (*)(const char *, int),
                       glob64_t *);
    real_glob64 = dlsym(RTLD_NEXT, "glob64");
    if (!real_glob64) {
      errno = ENOSYS;
      return -1;
    }
    return real_glob64(pattern, flags, errfunc, pglob);
  }
  if (strcmp(mock_mode, "notfound") == 0) {
    pglob->gl_pathc = 0;
    pglob->gl_pathv = NULL;
    return 0;
  }
  assert(strcmp(mock_mode, "exists") == 0);
  pglob->gl_pathc = 2;
  pglob->gl_pathv = mock_render_dirs;
  return 0;
}

void globfree(glob_t *pglob) {
  if (pglob->gl_pathv == mock_render_dirs) {
    pglob->gl_pathv = NULL;
    pglob->gl_pathc = 0;
    return;
  }
  // Default behavior: call real globfree
  void (*real_globfree)(glob_t *);
  real_globfree = dlsym(RTLD_NEXT, "globfree");
  if (!real_globfree) {
    errno = ENOSYS;
    return;
  }
  real_globfree(pglob);
}

void globfree64(glob64_t *pglob) {
  if (pglob->gl_pathv == mock_render_dirs) {
    pglob->gl_pathv = NULL;
    pglob->gl_pathc = 0;
    return;
  }
  void (*real_globfree64)(glob64_t *);
  real_globfree64 = dlsym(RTLD_NEXT, "globfree64");
  if (!real_globfree64) {
    errno = ENOSYS;
    return;
  }
  real_globfree64(pglob);
}

#define DUMMY_FD_128 128
#define DUMMY_FD_129 129

int mock_open_helper(const char *pathname) {
  const char *mock_mode = getenv("MOCK_OPEN_MODE");
  assert(mock_mode != NULL && "MOCK_OPEN_MODE environment variable is not set");
  if (strcmp(mock_mode, "deny") == 0) {
    errno = EACCES;
    return -1;
  }

  if (pathname == mock_render_dirs[0])
    return DUMMY_FD_128;

  if (pathname == mock_render_dirs[1]) {
    if (mock_mode && strcmp(mock_mode, "deny_second") == 0) {
      errno = EACCES;
      return -1;
    }
    return DUMMY_FD_129;
  }
  assert(0 && "Unexpected pathname in mock_open_helper");
}

int open(const char *pathname, int flags, ...) {
  if (pathname == mock_render_dirs[0] || pathname == mock_render_dirs[1])
    return mock_open_helper(pathname);

  // Call the real open function if not renderD12*.
  va_list ap;
  va_start(ap, flags);
  mode_t mode = va_arg(ap, mode_t);
  va_end(ap);
  int (*real_open)(const char *, int, ...);
  real_open = dlsym(RTLD_NEXT, "open");
  if (!real_open) {
    errno = ENOSYS;
    return -1;
  }
  return real_open(pathname, flags, mode);
}

int open64(const char *pathname, int flags, ...) {
  if (pathname == mock_render_dirs[0] || pathname == mock_render_dirs[1])
    return mock_open_helper(pathname);

  // Call the real open function if not renderD12*.
  va_list ap;
  va_start(ap, flags);
  mode_t mode = va_arg(ap, mode_t);
  va_end(ap);
  int (*real_open64)(const char *, int, ...);
  real_open64 = dlsym(RTLD_NEXT, "open64");
  if (!real_open64) {
    errno = ENOSYS;
    return -1;
  }
  return real_open64(pathname, flags, mode);
}

int close(int fd) {
  if (fd == DUMMY_FD_128 || fd == DUMMY_FD_129) {
    // Mock close for our dummy file descriptors.
    return 0;
  }

  int (*real_close)(int) = dlsym(RTLD_NEXT, "close");
  if (!real_close) {
    errno = ENOSYS;
    return -1;
  }
  return real_close(fd);
}
