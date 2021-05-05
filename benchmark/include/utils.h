#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "hnswlib.h"
#include "stopwatch.h"

#include <unordered_set>

using namespace std;
using namespace hnswlib;




inline bool exists_test(const std::string &name)
{
    ifstream f(name.c_str());
    return f.good();
}

/*****************************************************
 * Iterating over the elements of a pointer
 * Reference https://stackoverflow.com/questions/25060051/how-to-perform-a-range-based-c11-for-loop-on-char-argv/25060389#25060389
 *****************************************************/

template <typename T>
struct array_view
{
    T * first, * last;
    array_view(T * a, std::size_t n) : first(a), last(a + n) {}
    T * begin() const { return first; }
    T * end() const { return last; }
};

template <typename T>
array_view<T> view_array(T * a, std::size_t n)
{
    return array_view<T>(a, n);
}



/*****************************************************
 * I/O functions for fvecs and ivecs
 * Reference https://github.com/facebookresearch/faiss/blob/e86bf8cae1a0ecdaee1503121421ed262ecee98c/demos/demo_sift1M.cpp
 *****************************************************/

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}


// --------------------------------------------- int functions -------------------------------------------
static void get_gt(unsigned int *massQA, unsigned char *massQ, unsigned char *mass, size_t vecsize, size_t qsize, L2SpaceI &l2space,
                   size_t vecdim, vector<std::priority_queue<std::pair<int, labeltype>>> &answers, size_t k)
{

    (vector<std::priority_queue<std::pair<int, labeltype>>>(qsize)).swap(answers);
    DISTFUNC<int> fstdistfunc_ = l2space.get_dist_func();
    cout << qsize << "\n";
    for (int i = 0; i < qsize; i++)
    {
        for (int j = 0; j < k; j++)
        {
            answers[i].emplace(0.0f, massQA[1000 * i + j]);
        }
    }
}

static float test_approx(unsigned char *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<int> &appr_alg, size_t vecdim,
                         vector<std::priority_queue<std::pair<int, labeltype>>> &answers, size_t k)
{
    size_t correct = 0;
    size_t total = 0;
    //uncomment to test in parallel mode:
    //#pragma omp parallel for
    for (int i = 0; i < qsize; i++)
    {

        std::priority_queue<std::pair<int, labeltype>> result = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::priority_queue<std::pair<int, labeltype>> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (gt.size())
        {

            g.insert(gt.top().second);
            gt.pop();
        }

        while (result.size())
        {
            if (g.find(result.top().second) != g.end())
            {

                correct++;
            }
            else
            {
            }
            result.pop();
        }
    }
    return 1.0f * correct / total;
}

static void test_vs_recall(unsigned char *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<int> &appr_alg, size_t vecdim,
                           vector<std::priority_queue<std::pair<int, labeltype>>> &answers, size_t k)
{
    vector<size_t> efs; // = { 10,10,10,10,10 };
    for (int i = k; i < 30; i++)
    {
        efs.push_back(i);
    }
    for (int i = 30; i < 100; i += 10)
    {
        efs.push_back(i);
    }
    for (int i = 100; i < 500; i += 40)
    {
        efs.push_back(i);
    }
    for (size_t ef : efs)
    {
        appr_alg.setEf(ef);
        StopW stopw = StopW();

        float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;

        cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
        if (recall > 1.0)
        {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}

// --------------------------------------------- float functions -------------------------------------------
static void get_gt(unsigned int *massQA, float *massQ, float *mass, size_t vecsize, size_t qsize, L2Space &l2space, size_t vecdim,
       vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {

    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<float> fstdistfunc_ = l2space.get_dist_func();
    cout << qsize << "\n";
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            float other = fstdistfunc_(massQ + i * vecdim, mass + massQA[100 * i + j] * vecdim,
                                       l2space.get_dist_func_param());
            answers[i].emplace(other, massQA[100 * i + j]);
        }
    }
}

static float test_approx(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
                         vector<std::priority_queue<std::pair<float, labeltype>>> &answers, size_t k)
{
    size_t correct = 0;
    size_t total = 0;
    //uncomment to test in parallel mode:
    //#pragma omp parallel for
    for (int i = 0; i < qsize; i++)
    {

        std::priority_queue<std::pair<float, labeltype>> result = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::priority_queue<std::pair<float, labeltype>> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (gt.size())
        {

            g.insert(gt.top().second);
            gt.pop();
        }

        while (result.size())
        {
            if (g.find(result.top().second) != g.end())
            {

                correct++;
            }
            result.pop();
        }
    }
    return 1.0f * correct / total;
}

static void test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
                           vector<std::priority_queue<std::pair<float, labeltype>>> &answers, size_t k)
{
    vector<size_t> efs; // = { 10,10,10,10,10 };
    for (int i = k; i < 30; i++)
    {
        efs.push_back(i);
    }
    for (int i = 30; i < 100; i += 10)
    {
        efs.push_back(i);
    }
    for (int i = 100; i < 500; i += 40)
    {
        efs.push_back(i);
    }
    for (size_t ef : efs)
    {
        appr_alg.setEf(ef);
        StopW stopw = StopW();

        float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;

        cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
        if (recall > 1.0)
        {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}