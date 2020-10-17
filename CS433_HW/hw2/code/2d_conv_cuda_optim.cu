/*
This is the cuda version of 2d-convolution with multiple thread blocks.
In this problem, our output with size of [N, F, H_, W_];
So we divides the task into (H_ * W_) blocks, and each nlock has (N * F) threads.
*/
#include "matr_def.h"


__global__ void conv2d_cuda_optim_kernel(float *out_matr, float *fm_matr, float *kn_matr,
                                    int in_channel, int out_channel, int height, int width, 
                                    int ksize_x, int ksize_y);

void Conv2D_cuda_optim(Matrix &out, Matrix fm, Matrix kn) {
    fm.cuda(); kn.cuda(); out.cuda();
    dim3 block_sz(out.d1, out.d2);
    dim3 grid_sz(out.d3, out.d4);
    conv2d_cuda_optim_kernel<<<grid_sz,block_sz>>>(out.element, fm.element, kn.element, 
                                            kn.d2, kn.d1, fm.d3, fm.d4, kn.d3, kn.d4);
    out.cpu();
}


__global__ void conv2d_cuda_optim_kernel(float *out_matr, float *fm_matr, float *kn_matr,
                                    int in_channel, int out_channel, int height, int width, 
                                    int ksize_x, int ksize_y) {
    int batch_id = threadIdx.x, channel_id = threadIdx.y;
    int row = blockIdx.x, col = blockIdx.y;
    float cell_value = 0;
    for (int c = 0; c < in_channel; c++) // each in-channel
        for (int i = 0; i < ksize_x; i++) 
            for (int j = 0; j < ksize_y; j++) // each lacation of a kernel 
            cell_value += kn_matr[channel_id*in_channel*ksize_x*ksize_y + c*ksize_x*ksize_y + i*ksize_y + j] * 
                    fm_matr[batch_id*in_channel*height*width + c*height*width + (row+i)*width + (col+j)];
    // printf("[%d,%d,%d,%d] = %f\n", batch_id, channel_id, row, col, cell_value);
    out_matr[batch_id*out_channel*(height - ksize_x + 1)*(width - ksize_y + 1) + 
                channel_id*(height - ksize_x + 1)*(width - ksize_y + 1) +
                row*(width - ksize_y + 1) + col] = cell_value;
}

int main() {
    //Initialize Matrix
    Matrix fm(N, C, H, W), kn(F, C, K, K);
    Matrix out(N, F, H-K+1, W-K+1);
    Matrix truth(N, F, H-K+1, W-K+1);
    fm.fill_value(1.0);
    kn.fill_value(0.5);
    truth.fill_value(288.0);
    printf("The feature map is filled with %f;\n",*fm.get(1,2,3,4));
    printf("The kernel is filled with %f;\n",*kn.get(1,2,3,4));
    clock_t st,ed;
    st = clock();
    Conv2D_cuda_optim(out, fm, kn);
    ed = clock();
    printf("It takes %f ms to calculate the convolution...", (double)(ed-st)/CLOCKS_PER_SEC * 1000);
    if (out == truth)
        printf("Result is correct! (%f)\n", *out.get(1,2,3,4));
    else
        printf("Result is wrong! (%f)\n", *out.get(1,2,3,4));
}
