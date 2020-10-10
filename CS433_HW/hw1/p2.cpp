#include <stdlib.h>
#include <iostream>
#include <omp.h>

using namespace std;


template <typename T>
void Matrix_Mul(T** Mat_A, T** Mat_B, T** Mat_C, int m, int n, int k) {
    /* Assume all imputs are correct!
    * A(m*n)  B(n*k) => C(m*k) */

    // Calculate each cell of C
    // For example, (i,j) of C depend on i_th row of A and j_th column of B
    #pragma omp parallel num_threads(m*k)
    {
        int tmp = omp_get_thread_num();
        int i = tmp / k, j = tmp % k;

        // For each cell, we use reduction to sum results of dot-multiplication
        int cell_value = 0;
        #pragma omp parallel for reduction(+:cell_value)
        for (int idx = 0; idx < n; idx++)
            cell_value += Mat_A[i][idx] * Mat_B[idx][j];
        Mat_C[i][j] = cell_value;
       
    };

}


int main()
{
    int** A, ** B, ** C;
    int m, n, k;
    cout << "Input size m,n,k: ";
    cin >> m >> n >> k;
    // Input A and B:
    cout << "Input A: \n";
    A = new int* [m];
    for (int i = 0; i < m; i++) {
        A[i] = new int[n];
        for (int j = 0; j < n; j++)
            cin >> A[i][j];
    }
    cout << "Input B: \n";
    B = new int* [n];
    for (int i = 0; i < n; i++) {
        B[i] = new int[k];
        for (int j = 0; j < k; j++)
            cin >> B[i][j];
    }

    // Initialize Mat_C
    C = new int* [m];
    for (int i = 0; i < m; i++) 
        C[i] = new int[k];

    // Matrix multiplication
    Matrix_Mul(A, B, C, m, n, k);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++)
            cout << C[i][j] << ' ';
        cout << '\n';
    }
    
}

