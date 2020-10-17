#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

using namespace std;

//parallel
template <typename T>
T Vector_Sum(T* vec, int length) {
    T global_sum = 0;
    #pragma omp parallel for reduction(+:global_sum)
    for (int i = 0; i < length; i++)
        global_sum += vec[i];

    return global_sum;
}

//no parallel
template <typename T>
T Vector_Sum_Orig(T* vec, int length) {
    T global_sum = 0;
    for (int idx = 0; idx < length; idx++)
        global_sum += vec[idx];

    return global_sum;
}

int main()
{
    int sz, res;
    cin >> sz;
    int* vec = new int[sz];
    for (int i = 0; i < sz; i++)
        vec[i] = 1;

    //count time for each way
    clock_t st, ed;
    st = clock();
    res = Vector_Sum(vec, sz);
    ed = clock();
    cout << "Parallel: " << res << " (cost " << double(ed - st) << "ms);\n";


    st = clock();
    res = Vector_Sum_Orig(vec, sz);
    ed = clock();
    cout << "Pipeline: " << res << " (cost " << double(ed - st) << "ms);\n";
    return 0;
}