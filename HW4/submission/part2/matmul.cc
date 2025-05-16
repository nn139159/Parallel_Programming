#include <mpi.h>
#include <cstdlib>
#include <cstring>
void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    /* TODO: The data is stored in a_mat and b_mat.
     * You need to allocate memory for a_mat_ptr and b_mat_ptr,
     * and copy the data from a_mat and b_mat to a_mat_ptr and b_mat_ptr, respectively.
     * You can use any size and layout you want if they provide better performance.
     * Unambitiously copying the data is also acceptable.
     *
     * The matrix multiplication will be performed on a_mat_ptr and b_mat_ptr.
     */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate rows per process
    int rows_per_process = n / size;
    int remainder = n % size;

    int current_rows = rows_per_process + (rank < remainder ? 1 : 0);
    *a_mat_ptr = (int*)malloc(current_rows * m * sizeof(int));

    // Rank 0 sends rows of A to each process
    if (rank == 0) {
        int *send_counts = (int*)malloc(size * sizeof(int));
        int *displacements = (int*)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            int rows = rows_per_process + (i < remainder ? 1 : 0);
            send_counts[i] = rows * m;
            displacements[i] = offset;
            offset += send_counts[i];
        }

        MPI_Scatterv(a_mat, send_counts, displacements, MPI_INT, *a_mat_ptr, (current_rows * m), MPI_INT, 0, MPI_COMM_WORLD);

        free(send_counts);
        free(displacements);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, *a_mat_ptr, (current_rows * m), MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Broadcast B matrix (transposed for better performance)
    *b_mat_ptr = (int*)malloc(l * m * sizeof(int));
    if (rank == 0) {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < l; ++j)
                (*b_mat_ptr)[j * m + i] = b_mat[j * m + i]; // Already in transposed format
    }
    MPI_Bcast(*b_mat_ptr, (l * m), MPI_INT, 0, MPI_COMM_WORLD);
}

void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
{
    /* TODO: Perform matrix multiplication on a_mat and b_mat. Which are the matrices you've
     * constructed. The result should be stored in out_mat, which is a continuous memory placing n *
     * l elements of int. You need to make sure rank 0 receives the result.
     */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = n / size;
    int remainder = n % size;
    int current_rows = rows_per_process + (rank < remainder ? 1 : 0);

    // Local output buffer
    int *local_out = (int*)malloc(current_rows * l * sizeof(int));

    // Perform multiplication (b_mat is transposed)
    for (int i = 0; i < current_rows; ++i) {
        for (int j = 0; j < l; ++j) {
            int sum = 0;
            for (int k = 0; k < m; ++k)
                sum += a_mat[i * m + k] * b_mat[j * m + k];
            local_out[i * l + j] = sum;
        }
    }

    // Gather results to rank 0
    if (rank == 0) {
        int *receive_counts = (int*)malloc(size * sizeof(int));
        int *displacements = (int*)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            int rows = rows_per_process + (i < remainder ? 1 : 0);
            receive_counts[i] = rows * l;
            displacements[i] = offset;
            offset += receive_counts[i];
        }

        MPI_Gatherv(local_out, (current_rows * l), MPI_INT, out_mat, receive_counts, displacements, MPI_INT, 0, MPI_COMM_WORLD);

        free(receive_counts);
        free(displacements);
    } else {
        MPI_Gatherv(local_out, (current_rows * l), MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
    }

    free(local_out);
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    /* TODO */
    free(a_mat);
    free(b_mat);
}
