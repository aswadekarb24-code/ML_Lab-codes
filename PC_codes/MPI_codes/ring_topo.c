#include<stdio.h>
#include<mpi.h>

int main(int argc, char *argv[]){
    int my_rank, num_procs, value, sum=0;
    MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    if(my_rank==0){
        MPI_Send(&my_rank, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&sum, 1, MPI_INT, num_procs-1, 0, MPI_COMM_WORLD, &status);
        printf("My rank %d Final Sum %d\n", my_rank, sum);
    } 
    else{
        MPI_Recv(&value, 1, MPI_INT, my_rank-1, 0, MPI_COMM_WORLD, &status);
        sum=my_rank+value;
        int dest=(my_rank==num_procs-1) ? 0:my_rank+1;
        MPI_Send(&sum, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
