//  Windows
#ifndef TIMERS_H
#define TIMERS_H

#if __STDC_VERSION__ >= 199901L
#define _XOPEN_SOURCE 600
#else
#define _XOPEN_SOURCE 500
#endif /* __STDC_VERSION__ */

#include <time.h>
#include <sys/time.h>
#include <time.h>
#include <stddef.h>
#include <stdio.h>

double get_wall_time();
double get_cpu_time();

struct timespec diff(struct timespec start, struct timespec end);

#endif //TIMERS_H
