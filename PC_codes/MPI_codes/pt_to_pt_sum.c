#include<stdio.h>
#include<mpi.h>

int main(int argc, char *argv[]){
    int my_rank, num_procs, value, sum=0, i;
    MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    if(my_rank==0){
        for(i=1; i<num_procs; i++){
            MPI_Recv(&value, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            sum+=value;
            printf("My rank = %d, Sum = %d\n", my_rank, sum);
        }
    } else{
        MPI_Send(&my_rank,1,MPI_INT, 0,0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}