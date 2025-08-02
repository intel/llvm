#define _GNU_SOURCE

#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <glob.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

int glob(const char *pattern, int flags, int (*errfunc)(const char *, int),
         glob_t *pglob) {
  const char *mock_mode = getenv("MOCK_GLOB_MODE");
  if (mock_mode && strcmp(mock_mode, "exists") == 0) {
    pglob->gl_pathc = 2;
    pglob->gl_pathv = malloc(2 * sizeof(char *));
    pglob->gl_pathv[0] = strdup("/dev/dri/renderD128");
    pglob->gl_pathv[1] = strdup("/dev/dri/renderD129");
    return 0;
  }
  // Default behavior: no matches
  pglob->gl_pathc = 0;
  pglob->gl_pathv = NULL;
  return 0;
}

int glob64(const char *pattern, int flags, int (*errfunc)(const char *, int),
           glob64_t *pglob) {
  const char *mock_mode = getenv("MOCK_GLOB_MODE");
  if (mock_mode && strcmp(mock_mode, "exists") == 0) {
    pglob->gl_pathc = 2;
    pglob->gl_pathv = malloc(2 * sizeof(char *));
    pglob->gl_pathv[0] = strdup("/dev/dri/renderD128");
    pglob->gl_pathv[1] = strdup("/dev/dri/renderD129");
    return 0;
  }
  // Default behavior: no matches
  pglob->gl_pathc = 0;
  pglob->gl_pathv = NULL;
  return 0;
}

void globfree(glob_t *pglob) {
  if (pglob->gl_pathv) {
    for (size_t i = 0; i < pglob->gl_pathc; ++i) {
      free(pglob->gl_pathv[i]);
    }
    free(pglob->gl_pathv);
    pglob->gl_pathv = NULL;
    pglob->gl_pathc = 0;
  }
}

void globfree64(glob64_t *pglob) {
  if (pglob->gl_pathv) {
    for (size_t i = 0; i < pglob->gl_pathc; ++i) {
      free(pglob->gl_pathv[i]);
    }
    free(pglob->gl_pathv);
    pglob->gl_pathv = NULL;
    pglob->gl_pathc = 0;
  }
}

static int (*real_open)(const char *, int, ...) = NULL;
static int (*real_open64)(const char *, int, ...) = NULL;

#define DUMMY_FD_128 128
#define DUMMY_FD_129 129

int mock_open_helper(const char *pathname) {
  const char *mock_mode = getenv("MOCK_OPEN_MODE");
  assert(mock_mode != NULL && "MOCK_OPEN_MODE environment variable is not set");
  if (strcmp(mock_mode, "deny") == 0) {
    errno = EACCES;
    return -1;
  }

  if (strstr(pathname, "renderD128"))
    return DUMMY_FD_128;

  if (strstr(pathname, "renderD129")) {
    if (mock_mode && strcmp(mock_mode, "deny_second") == 0) {
      errno = EACCES;
      return -1;
    }
    return DUMMY_FD_129;
  }
  assert(0 && "Unexpected pathname in mock_open_helper");
}

int open(const char *pathname, int flags, ...) {
  if (strstr(pathname, "renderD12"))
    return mock_open_helper(pathname);

  // Call the real open function if not renderD12*.
  va_list ap;
  va_start(ap, flags);
  mode_t mode = va_arg(ap, mode_t);
  va_end(ap);
  if (!real_open)
    real_open = dlsym(RTLD_NEXT, "open");
  return real_open(pathname, flags, mode);
}

int open64(const char *pathname, int flags, ...) {
  if (strstr(pathname, "renderD12"))
    return mock_open_helper(pathname);

  // Call the real open function if not renderD12*.
  va_list ap;
  va_start(ap, flags);
  mode_t mode = va_arg(ap, mode_t);
  va_end(ap);
  if (!real_open64)
    real_open64 = dlsym(RTLD_NEXT, "open64");
  return real_open64(pathname, flags, mode);
}

static int (*real_close)(int) = NULL;

int close(int fd) {
  if (fd == DUMMY_FD_128 || fd == DUMMY_FD_129) {
    // Mock close for our dummy file descriptors.
    return 0;
  }

  int (*real_close)(int) = dlsym(RTLD_NEXT, "close");
  return real_close(fd);
}
