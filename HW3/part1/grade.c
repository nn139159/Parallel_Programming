#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>

#include "cg_impl.h"
#include "globals.h"
#include "timers.h"

void reference_init(double *zeta);
void reference_iterate(double *zeta, int *it);
void default_init(double *zeta);
void default_iterate(double *zeta, int *it);

void print_scores(double stu_time, double ref_time, bool verified)
{
    double max_score = 30;
    double max_perf_score = 0.8 * max_score;
    double correctness_score = 0.2 * max_score;
    correctness_score = (verified == true) ? correctness_score : 0;

    double ratio = (ref_time / stu_time);

    double slope = max_perf_score / (0.7 - 0.3);
    double offset = 0.3 * slope;

    double perf_score = (verified == true) ? (ratio * slope) - offset : 0;

    if (perf_score < 0)
        perf_score = 0;
    if (perf_score > max_perf_score)
        perf_score = max_perf_score;

    printf("correctness : %lf\n", correctness_score);
    printf("performance : %lf\n", perf_score);
    printf("total       : %lf\n", correctness_score + perf_score);
}

int main(int argc, char *argv[])
{
    int num_threads = omp_get_max_threads();
    double zeta;
    double t, t_total = 0, reference_total = 0, default_total = 0;
    bool verified;
    double zeta_verify_value, epsilon, err;
    char *t_names[T_LAST];

    omp_set_num_threads(num_threads);
    zeta_verify_value = VALID_RESULT;

    printf("\nCG start...\n\n");
    printf(" Size: %11d\n", NA);
    printf(" Iterations: %5d\n", NITER);
    printf(" Running with %d threads\n", num_threads);
    printf("\n");

    for (int i = 0; i < T_LAST; i++)
    {
        timer_clear(i);
    }
    timer_start(T_INIT);
    init(&zeta);
    zeta = 0.0;
    for (int it = 1; it <= 1; it++)
    {
        iterate(&zeta, &it);
    }
    for (int i = 0; i < NA + 1; i++)
    {
        x[i] = 1.0;
    }
    zeta = 0.0;
    timer_stop(T_INIT);
    t_total += timer_read(T_INIT);

    timer_start(T_BENCH);
    for (int it = 1; it <= NITER; it++)
    {
        iterate(&zeta, &it);
    }
    timer_stop(T_BENCH);
    t = timer_read(T_BENCH);
    t_total += t;

    epsilon = 1.0e-10;
    err = fabs(zeta - zeta_verify_value) / zeta_verify_value;
    if (err <= epsilon)
    {
        verified = true;
        printf(" VERIFICATION SUCCESSFUL\n");
        printf(" Zeta is    %20.13E\n", zeta);
        printf(" Error is   %20.13E\n", err);
    }
    else
    {
        verified = false;
        printf(" VERIFICATION FAILED\n");
        printf(" Zeta                %20.13E\n", zeta);
        printf(" The correct zeta is %20.13E\n", zeta_verify_value);
    }

    for (int i = 0; i < T_LAST; i++)
    {
        timer_clear(i);
    }
    timer_start(T_INIT);
    reference_init(&zeta);
    zeta = 0.0;
    for (int it = 1; it <= 1; it++)
    {
        reference_iterate(&zeta, &it);
    }
    for (int i = 0; i < NA + 1; i++)
    {
        x[i] = 1.0;
    }
    zeta = 0.0;
    timer_stop(T_INIT);
    reference_total += timer_read(T_INIT);

    timer_start(T_BENCH);
    for (int it = 1; it <= NITER; it++)
    {
        reference_iterate(&zeta, &it);
    }
    timer_stop(T_BENCH);
    t = timer_read(T_BENCH);
    reference_total += t;

    for (int i = 0; i < T_LAST; i++)
    {
        timer_clear(i);
    }
    timer_start(T_INIT);
    default_init(&zeta);
    zeta = 0.0;
    for (int it = 1; it <= 1; it++)
    {
        default_iterate(&zeta, &it);
    }
    for (int i = 0; i < NA + 1; i++)
    {
        x[i] = 1.0;
    }
    zeta = 0.0;
    timer_stop(T_INIT);
    default_total += timer_read(T_INIT);

    timer_start(T_BENCH);
    for (int it = 1; it <= NITER; it++)
    {
        default_iterate(&zeta, &it);
    }
    timer_stop(T_BENCH);
    t = timer_read(T_BENCH);
    default_total += t;

    printf("\nreference time : %lfs\n", reference_total);
    printf("default time   : %lfs\n", default_total);
    printf("student time   : %lfs\n\n", t_total);

    if (default_total - 0.1 < t_total)
    {
        printf("Your implementation should be faster than default - 0.1s!\n\n");
        verified = false;
    }

    print_scores(t_total, reference_total, verified);

    return 0;
}
