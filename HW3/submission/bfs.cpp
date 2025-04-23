#include "bfs.h"

#include <cstdlib>
#include <omp.h>
#include <vector>
#include "../common/graph.h"

#ifdef VERBOSE
#include "../common/CycleTimer.h"
#include <stdio.h>
#endif // VERBOSE

#define bitmap_en
constexpr int ROOT_NODE_ID = 0;
constexpr int NOT_VISITED_MARKER = -1;

void vertex_set_clear(VertexSet *list)
{
    list->count = 0;
}

void vertex_set_init(VertexSet *list, int count)
{
    list->max_vertices = count;
    list->vertices = new int[list->max_vertices];
    vertex_set_clear(list);
}

void vertex_set_destroy(VertexSet *list)
{
    delete[] list->vertices;
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances)
{/*
    for (int i = 0; i < frontier->count; i++)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                distances[outgoing] = distances[node] + 1;
                int index = new_frontier->count++;
                new_frontier->vertices[index] = outgoing;
            }
        }
    }*/
    
    std::vector<std::vector<int>> local_frontiers(omp_get_max_threads());
    #pragma omp parallel
    {
	std::vector<int>& local = local_frontiers[omp_get_thread_num()];
        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < frontier->count; i++)
        {
            int node = frontier->vertices[i];
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];

                if (distances[outgoing] == NOT_VISITED_MARKER)
                {
		    distances[outgoing] = distances[node] + 1;
                    local.push_back(outgoing);
                }
            }
        }
    }

    for (auto& local : local_frontiers)
    {
        for (int v : local)
        {
            new_frontier->vertices[new_frontier->count++] = v;
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    // free memory
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

#ifndef bitmap_en
bool bottom_up_step(Graph g, VertexSet *new_frontier, int *distances, int current_level)
{
    bool changed = false;
    std::vector<std::vector<int>> local_frontiers(omp_get_max_threads());
    #pragma omp parallel
    {
        std::vector<int> &local = local_frontiers[omp_get_thread_num()];
        bool local_changed = false;

        #pragma omp for schedule(dynamic, 1024)
        for (int node = 0; node < g->num_nodes; node++)
        {
            if (distances[node] != NOT_VISITED_MARKER) continue;

            int start = g->incoming_starts[node];
            int end = (node == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[node + 1];

            for (int edge = start; edge < end; edge++)
            {
                int neighbor = g->incoming_edges[edge];
                if (distances[neighbor] == current_level - 1)
                {
                    distances[node] = current_level;
                    local.push_back(node);
                    local_changed = true;
                    break;
                }
            }
        }

        if (local_changed)
            #pragma omp atomic write
            changed = true;
    }

    for (const auto &local : local_frontiers)
    {
        for (int v : local)
            new_frontier->vertices[new_frontier->count++] = v;
    }

    return changed;
}
#else
struct bitmap_t {
    int count;
    int size; // = ceil(num_nodes / 64.0)
    uint64_t *bitmap;
};

inline void bitmap_clear(bitmap_t *bitmap)
{
    for (int i = 0; i < bitmap->size; ++i)
    {
        bitmap->bitmap[i] = 0;
    }
    bitmap->count = 0;
}

void bitmap_init(bitmap_t *bitmap, int bitcount)
{
    bitmap->size = (bitcount + 63) / 64; 
    bitmap->bitmap = (uint64_t *)malloc(sizeof(uint64_t) * bitmap->size);

    bitmap_clear(bitmap);
}

void bitmap_release(bitmap_t *bitmap)
{
    if (bitmap->bitmap != nullptr) {
        free(bitmap->bitmap);
        bitmap->bitmap = nullptr;
    }
    bitmap->size = 0;
    bitmap->count = 0;
}

inline uint64_t bitmap_get(bitmap_t *bitmap, int bit) {
    return (bitmap->bitmap[bit >> 6] >> (bit & 0x3f)) & 1;
}

inline void bitmap_set(bitmap_t *bitmap, int bit) {
    bitmap->count += 1;
    bitmap->bitmap[bit >> 6] |= (uint64_t)1 << (bit & 0x3f);
}

void bottom_up_step(Graph g, bitmap_t *frontier, bitmap_t *new_frontier, int *distances, int level)
{
    int num_nodes = g->num_nodes;
    int *incoming_starts = g->incoming_starts;
    Vertex *incoming_edges = g->incoming_edges;

    #pragma omp parallel
    {
        std::vector<int> local_frontier;
        #pragma omp for schedule(dynamic, 1024)
        for (int v = 0; v < num_nodes; ++v)
        {
            if (distances[v] != NOT_VISITED_MARKER) continue;

            int start_edge = incoming_starts[v];
            int end_edge = (v == num_nodes - 1) ? g->num_edges : incoming_starts[v + 1];

            for (int edge = start_edge; edge < end_edge; ++edge)
            {
                int parent_v = incoming_edges[edge];
                if (bitmap_get(frontier, parent_v)) 
                {
                    distances[v] = level;
                    local_frontier.push_back(v);
                    break;
                }
            }
        }

        #pragma omp critical
        for (int v : local_frontier)
        {
            bitmap_set(new_frontier, v);
            #pragma omp atomic
            new_frontier->count++;
        }
    }
}
#endif

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
#ifndef bitmap_en
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    frontier->vertices[0] = ROOT_NODE_ID;
    frontier->count = 1;
#else
    bitmap_t bitmap1;
    bitmap_t bitmap2;
    bitmap_init(&bitmap1, graph->num_nodes);
    bitmap_init(&bitmap2, graph->num_nodes);

    bitmap_t *frontier = &bitmap1;
    bitmap_t *new_frontier = &bitmap2;

    bitmap_set(frontier, ROOT_NODE_ID);
#endif

    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;
    
    sol->distances[ROOT_NODE_ID] = 0;

    int level = 1;
    bool changed = true;
    while (changed) {
#ifndef bitmap_en
        vertex_set_clear(new_frontier);

        changed = bottom_up_step(graph, new_frontier, sol->distances, level);

        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
#else
        bitmap_clear(new_frontier);

        bottom_up_step(graph, frontier, new_frontier, sol->distances, level);

        changed = new_frontier->count > 0;

        bitmap_t *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
#endif
        level++;
    }

    // Release memory
#ifndef bitmap_en
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
#else
    bitmap_release(&bitmap1);
    bitmap_release(&bitmap2);
#endif
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    const float THRESHOLD = 0.05f;
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;
#ifdef bitmap_en
    bitmap_t bitmap1;
    bitmap_t bitmap2;
    bitmap_init(&bitmap1, graph->num_nodes);
    bitmap_init(&bitmap2, graph->num_nodes);

    bitmap_t *bitmap_frontier = &bitmap1;
    bitmap_t *new_bitmap_frontier = &bitmap2;
#endif
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[0] = ROOT_NODE_ID;
    frontier->count = 1;
    sol->distances[ROOT_NODE_ID] = 0;

    int level = 1;
    bool use_bottom_up = false;

    while (frontier->count > 0)
    {
        vertex_set_clear(new_frontier);

        float frontier_density = (float)frontier->count / graph->num_nodes;

        if (frontier_density > THRESHOLD) {
#ifndef bitmap_en
            bottom_up_step(graph, new_frontier, sol->distances, level);
#else
            bitmap_clear(bitmap_frontier);
            for (int i = 0; i < frontier->count; i++) {
                bitmap_set(bitmap_frontier, frontier->vertices[i]);
            }
            bitmap_clear(new_bitmap_frontier);
	    new_bitmap_frontier->count = 0;

            bottom_up_step(graph, bitmap_frontier, new_bitmap_frontier, sol->distances, level);

            frontier->count = 0;
            for (int i = 0; i < graph->num_nodes; i++) {
                if (bitmap_get(new_bitmap_frontier, i))
                    frontier->vertices[frontier->count++] = i;
            }
#endif
        }
        else {
            top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef bitmap_en
            VertexSet *tmp = frontier;
            frontier = new_frontier;
            new_frontier = tmp;
#endif
        }
#ifndef bitmap_en
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
#endif
        level++;
    }

    vertex_set_destroy(frontier);
    vertex_set_destroy(new_frontier);
#ifdef bitmap_en
    bitmap_release(bitmap_frontier);
    bitmap_release(new_bitmap_frontier);
#endif
}

