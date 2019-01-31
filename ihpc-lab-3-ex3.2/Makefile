CC = mpicc

FP_LIBS = -lm
GD_LIBS = -lm

FP_OBJS = MPI_Fempois.o
GD_OBJS = GridDist.o

all: MPI_Fempois GridDist

clean:
	rm -f *.o 

MPI_Fempois: $(FP_OBJS)
	mpicc -o $@.out $(FP_OBJS) $(FP_LIBS)

GridDist: $(GD_OBJS)
	gcc -o $@.out $(GD_OBJS) $(GD_LIBS)

MPI_Fempois.o:
	mpicc -c MPI_Fempois.c

GridDist.o:
	gcc -c GridDist.c



