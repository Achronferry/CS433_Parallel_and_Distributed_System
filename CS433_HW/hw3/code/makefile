P1=problem1
P2=problem2
MPICC?=mpicc

all: ${P1}.c ${P2}.c 

p1: ${P1}.c
	${MPICC} -o ${P1} ${P1}.c

p2: ${P2}.c
	${MPICC} -o ${P2} ${P2}.c

clean:
	rm -f ${P1}
	rm -f ${P2}
