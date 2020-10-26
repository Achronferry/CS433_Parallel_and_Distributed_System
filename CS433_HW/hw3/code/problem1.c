#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#define MAX_SIZE 128

int main(int argc, char* argv[])
{
    char beacon_packets[MAX_SIZE];  
    char ACK[MAX_SIZE];  
    memset(ACK, '\0', sizeof(ACK));

    int my_rank; // 记录当前进程号 (0-sender, 1-receiver)           
    double st, ed; // 记录开始/结束时间
    
    // 初始化MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // 发送者（0）
    if (my_rank == 0) {
        strcpy(beacon_packets, "abcdefghij"); // 生成beacon packets
        st = MPI_Wtime(); // 开始计时
        /* 将字符串发送给进程1; */
        printf("[Process %d] Sending \"%s\" to process 1;\n", my_rank, beacon_packets);
        MPI_Send(beacon_packets, strlen(beacon_packets)+1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        /* 接收来自进程1的回复; */
        MPI_Recv(ACK, MAX_SIZE, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Process %d] Received %ld ACKs;\n", my_rank, strlen(ACK));
        ed = MPI_Wtime(); // 停止计时;
        printf("[Process %d] Round trip cost time:%e \n", my_rank, ed-st);
    }
    // 接收者（1）
    else if (my_rank == 1) {
        /* 从进程0接收字符串; */
        MPI_Recv(beacon_packets, MAX_SIZE, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Process %d] Receiving \"%s\" from process 0;\n", my_rank, beacon_packets);
        /* 生成等长的ACK并回复; */
        for (int i = 0; i < strlen(beacon_packets); i++) 
            ACK[i] = '1';
        MPI_Send(ACK, strlen(ACK)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    // 结束MPI
    MPI_Finalize();
    return 0;

}

