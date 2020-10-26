#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>

#define MAX_SIZE 128

bool is_space(char c) {
    return (c == ' ' || c == '\t');
}

int main(int argc, char* argv[])
{
    char sentence[MAX_SIZE];
    char *left_stn = sentence;
    bool *end_flag = malloc(sizeof(bool)); //是否结束
    int my_rank, comm_sz;
    
    // 初始化MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); // 获取进程数；
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //获取当前进程号；

    int prev_proc = (my_rank - 1 + comm_sz) % comm_sz;
    int next_proc = (my_rank + 1) % comm_sz; //循环传递

    /* 进程0读取待传递的句子并传给进程1 */
    if (my_rank == 0) { 
        printf("[Process %d] Input a sentence: \n", my_rank);
        scanf("%[^\n]", left_stn);
        printf("[Process %d] Complete sentence:%s \n", my_rank, left_stn);
        *end_flag = false;
        MPI_Send(end_flag, 1, MPI_INT, next_proc, 0, MPI_COMM_WORLD); 
        MPI_Send(left_stn, strlen(left_stn)+1, MPI_CHAR, next_proc, 0, MPI_COMM_WORLD); 
    } 

    while (true) {
        /* 等待下次轮到该进程时进行接收, 对于end_flag设置其来源为任意进程，使得结束时不需要再遍历一轮 */
        MPI_Recv(end_flag, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (*end_flag) {
            MPI_Send(end_flag, 1, MPI_INT, next_proc, 0, MPI_COMM_WORLD); //告知下一个进程结束循环
            break;
        }
        // 当 end_flag 为 false 时，才会接着接收句子
        MPI_Recv(left_stn, MAX_SIZE, MPI_CHAR, prev_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* 最后一个进程输出eos并广播所有进程，结束循环 */
        if (*left_stn == '\0') {
            printf("[Process %d] Received: <eos>\n", my_rank);
            *end_flag = true;
            for (int i=0; i< comm_sz; i++)
                MPI_Send(end_flag, 1, MPI_INT, i, 0, MPI_COMM_WORLD); // 进行广播
            break;
        }

        /* 输出读取到的句子的下一个单词 */
        printf("[Process %d] Received: %s <--> Write word: ", my_rank, left_stn);
        while (!is_space(*left_stn) && *left_stn != '\0') {
            printf("%c", *left_stn);  
            left_stn++ ;
        }
        printf("\n");

        /* 跳过空白(space、tab) */
        while (is_space(*left_stn)) 
            left_stn++ ;

        /* 传递给下一个进程 */
        MPI_Send(end_flag, 1, MPI_INT, next_proc, 0, MPI_COMM_WORLD); //先告知是否结束（未结束）
        MPI_Send(left_stn, strlen(left_stn)+1, MPI_CHAR, next_proc, 0, MPI_COMM_WORLD); //发送剩下的句子

    }

    // 结束MPI
    MPI_Finalize();
    return 0;
}

