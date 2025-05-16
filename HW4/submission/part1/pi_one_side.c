#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Each process calculates its number of tosses
    long long int local_tosses = (world_rank == world_size - 1) ? (tosses / world_size) + (tosses % world_size) : (tosses / world_size);
    long long int local_count = 0;

    // Each process performs its own Monte Carlo simulation
    unsigned int seed = (unsigned int)(time(NULL)) + world_rank;
    for (long long int i = 0; i < local_tosses; i++)
    {
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;
        if (x * x + y * y <= 1.0)
        {
            local_count++;
        }
    }

    long long int *total_count_ptr;

    if (world_rank == 0)
    {
        // Master
	// Allocate memory and create a window
	MPI_Alloc_mem(sizeof(long long int), MPI_INFO_NULL, (void *)&total_count_ptr);
	*total_count_ptr = 0;

        MPI_Win_create(total_count_ptr, sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }
    else
    {
        // Workers
	// Other ranks do not disclose memory
	MPI_Win_create(NULL, 0, sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }

    // Make sure all processes have established windows before synchronization
    MPI_Barrier(MPI_COMM_WORLD);

    // Total hit counts for all workers (including rank 0)
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);

    MPI_Accumulate(&local_count, 1, MPI_LONG_LONG, 0, 0, 1, MPI_LONG_LONG, MPI_SUM, win);

    MPI_Win_unlock(0, win);

    // Synchronize to ensure accumulate completes
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = 4.0 * (double)(*total_count_ptr) / (double)tosses;

        MPI_Free_mem(total_count_ptr);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
