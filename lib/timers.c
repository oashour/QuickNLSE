//  Windows
#include "timers.h"

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL))
	{
        printf("Can't get time. Exiting\n");
        return 1;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

double get_cpu_time()
{
    return (double)clock() / CLOCKS_PER_SEC;
}

