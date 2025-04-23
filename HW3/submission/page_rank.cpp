#include "page_rank.h"

#include <cmath>
#include <cstdlib>
#include <omp.h>

#include "../common/graph.h"

// page_rank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void page_rank(Graph g, double *solution, double damping, double convergence)
{

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs
    /*
    int nnodes = num_nodes(g);
    double equal_prob = 1.0 / nnodes;
    for (int i = 0; i < nnodes; ++i)
    {
        solution[i] = equal_prob;
    }
    */
    /*
       For PP students: Implement the page rank algorithm here.  You
       are expected to parallelize the algorithm using openMP.  Your
       solution may need to allocate (and free) temporary arrays.

       Basic page rank pseudocode is provided below to get you started:

       // initialization: see example code above
       score_old[vi] = 1/nnodes;

       while (!converged) {

         // compute score_new[vi] for all nodes vi:
         score_new[vi] = sum over all nodes vj reachable from incoming edges
                            { score_old[vj] / number of edges leaving vj  }
         score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / nnodes;

         score_new[vi] += sum over all nodes v in graph with no outgoing edges
                            { damping * score_old[v] / nnodes }

         // compute how much per-node scores have changed
         // quit once algorithm has converged

         global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
         converged = (global_diff < convergence)
       }

     */
    int nnodes = num_nodes(g);
    double base_score = (1.0 - damping) / nnodes;


    // Allocate temporary arrays
    double *score_old = (double *)malloc(sizeof(double) * nnodes);
    double *score_new = (double *)malloc(sizeof(double) * nnodes);

    // Initialize scores to equal probability
    double init_score = 1.0 / nnodes;
    #pragma omp parallel for
    for (int i = 0; i < nnodes; ++i)
    {
        solution[i] = init_score;
        score_old[i] = init_score;
    }

    bool converged = false;

    while (!converged)
    {
        double global_diff = 0.0;

        // Handle dangling nodes (no outgoing edges)
        double dangling_sum = 0.0;
        #pragma omp parallel for reduction(+:dangling_sum)
        for (int i = 0; i < nnodes; ++i)
        {
            if (outgoing_size(g, i) == 0)
            {
                dangling_sum += score_old[i];
            }
        }

        // Calculate new scores
        #pragma omp parallel for
        for (int i = 0; i < nnodes; ++i)
        {
            double incoming_sum = 0.0;
            const Vertex *start = incoming_begin(g, i);
            const Vertex *end = incoming_end(g, i);

            for (const Vertex *vj = start; vj < end; ++vj)
            {
                int neighbor = *vj;
                int out_degree = outgoing_size(g, neighbor);
                if (out_degree > 0)
                {
                    incoming_sum += score_old[neighbor] / out_degree;
                }
            }

            score_new[i] = base_score + damping * (incoming_sum + dangling_sum / nnodes);
        }

        // Check convergence
        global_diff = 0.0;
        #pragma omp parallel for reduction(+:global_diff)
        for (int i = 0; i < nnodes; ++i)
        {
            global_diff += fabs(score_new[i] - score_old[i]);
        }

        converged = (global_diff < convergence);

        // Prepare for next iteration
        #pragma omp parallel for
        for (int i = 0; i < nnodes; ++i)
        {
            score_old[i] = score_new[i];
        }
    }

    // Copy final result to solution array
    #pragma omp parallel for
    for (int i = 0; i < nnodes; ++i)
    {
        solution[i] = score_new[i];
    }

    // Clean up
    free(score_old);
    free(score_new);
}
