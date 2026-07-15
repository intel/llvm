# Local GDB configuration to suppress glibc source warnings
set pagination off
set confirm off
set auto-load safe-path /

# Skip stepping into system libraries
skip -gfi /build/glibc*
skip -gfi */csu/*
skip -gfi */sysdeps/*
skip -gfi */nptl/*
skip file libc_start_call_main.h
skip file libc-start.c

# Don't break on entry to system functions
set scheduler-locking off
