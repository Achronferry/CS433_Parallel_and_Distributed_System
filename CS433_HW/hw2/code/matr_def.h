#ifndef MATR_DEF
#define MATR_DEF

#include <stdio.h>
#include <iostream>
#include <ctime>
#include "device_launch_parameters.h"

using namespace std;
/* 
feature map [N, C, H, W] * kernel  [F, C, K, K]
=> output [N, F, H-K+1, W-K+1]
*/
#define N 8
#define C 64
#define H 128
#define W 128
#define F 128
#define K 3

class Matrix {
public:
    int d1,d2,d3,d4;
    float *element;
    Matrix(int a, int b, int c, int d) {
        d1 = a; d2 = b;
        d3 = c; d4 = d;
        element = (float*)malloc(sizeof(float) * (d1*d2*d3*d4));
    }

    void fill_value(float v) {
        for (int i = 0; i < ( d1 * d2 * d3 * d4) ; i++)
            element[i] = v;
    }

    float* get(int a, int b, int c, int d) {
        return &element[a*d2*d3*d4 + b*d3*d4 + c*d4 + d];
    }

    void cuda() {    
        float *element_cuda;
        cudaMalloc((void **) &element_cuda, sizeof(float) * (d1*d2*d3*d4));
        cudaMemcpy(element_cuda, element, sizeof(float) * (d1*d2*d3*d4), cudaMemcpyHostToDevice);
        free(element);
        element = element_cuda;
    }

    void cpu() {    
        float *element_cpu = (float*)malloc(sizeof(float) * (d1*d2*d3*d4));
        cudaMemcpy(element_cpu, element, sizeof(float) * (d1*d2*d3*d4), cudaMemcpyDeviceToHost);
        cudaFree(element);
        element = element_cpu;
    }

    bool operator==(Matrix &t) const {
        if (d1 != t.d1 || d2 != t.d2 || d3 != t.d3 || d4 != t.d4)
            return false;
        for (int i = 0; i <d1*d2*d3*d4; i++)
            if (element[i] != t.element[i])
                return false;
        return true;
    }
};

#endif