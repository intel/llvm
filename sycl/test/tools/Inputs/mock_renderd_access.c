#define _GNU_SOURCE

#include <errno.h>
#include <glob.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

int glob(const char *pattern, int flags, int (*errfunc)(const char *, int),
         glob_t *pglob) {
  const char *mock_mode = getenv("MOCK_GLOB_MODE");
  if (mock_mode && strcmp(mock_mode, "exists") == 0) {
    // Simulate that /dev/dri/renderD* exists
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

int open(const char *pathname, int flags, ...) {
  const char *mock_mode = getenv("MOCK_OPEN_MODE");
  if (strstr(pathname, "renderD12")) {
    if (mock_mode && strcmp(mock_mode, "deny") == 0) {
      errno = EACCES;
      return -1;
    }

    if (mock_mode && strcmp(mock_mode, "deny_second") == 0) {
      // Simulate that the second file is not accessible
      if (strstr(pathname, "renderD129")) {
        errno = EACCES;
        return -1;
      } else {
        return 3;
      }
    }

    if (mock_mode && strcmp(mock_mode, "allow") == 0) {
      return 3; // Dummy fd
    }
  }
  // Default: permission denied
  errno = EACCES;
  return -1;
}

int close(int fd) { return 0; }
