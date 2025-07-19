#define _GNU_SOURCE

#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

int stat(const char *pathname, struct stat *statbuf) {
  const char *mock_mode = getenv("MOCK_STAT_MODE");
  if (strstr(pathname, "renderD128")) {
    if (mock_mode && strcmp(mock_mode, "notfound") == 0) {
      errno = ENOENT;
      return -1;
    }
    if (mock_mode && strcmp(mock_mode, "exists") == 0) {
      memset(statbuf, 0, sizeof(*statbuf));
      statbuf->st_mode = S_IFCHR | 0666;
      return 0;
    }
  }
  // Default: file does not exist
  errno = ENOENT;
  return -1;
}

int open(const char *pathname, int flags, ...) {
  const char *mock_mode = getenv("MOCK_OPEN_MODE");
  if (strstr(pathname, "renderD128")) {
    if (mock_mode && strcmp(mock_mode, "deny") == 0) {
      errno = EACCES;
      return -1;
    }
    if (mock_mode && strcmp(mock_mode, "allow") == 0) {
      return 3; // Dummy fd
    }
  }
  // Default: permission denied
  errno = EACCES;
  return -1;
}
