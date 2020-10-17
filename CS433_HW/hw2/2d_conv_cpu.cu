/*
This is the cpu version of 2d-convolution.
*/
#include "matr_def.h"

void Conv2D_cpu(Matrix &out, Matrix fm, Matrix kn) {
    out.fill_value(0);
    for (int bsize = 0; bsize < fm.d1; bsize++)
        for (int f = 0; f < kn.d1; f++)
            for (int c = 0; c < kn.d2; c++)
                for (int h_ = 0; h_ < fm.d3 - kn.d3 + 1; h_++)
                    for (int w_ = 0; w_ < fm.d4 - kn.d4 + 1; w_++)
                        for (int i = 0; i < kn.d3; i++)
                            for (int j = 0; j < kn.d4; j++)
                                *out.get(bsize,f,h_,w_) += 
                                *kn.get(f, c, i, j) * *fm.get(bsize, c, h_+i, w_+j);

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
    Conv2D_cpu(out, fm, kn);
    ed = clock();
    printf("It takes %f ms to calculate the convolution...", (double)(ed-st)/CLOCKS_PER_SEC * 1000);
    if (out == truth)
        printf("Result is correct! (%f)\n", *out.get(1,2,3,4));
    else
        printf("Result is wrong! (%f)\n", *out.get(1,2,3,4));
}
