
CC=gcc
LEX=flex
NVCC=nvcc
INC_D=include/
SRC_D=source/
LEX_D=lex/
LIBS= -L/usr/local/cuda/lib64 -lcuda -lcudart -lfl -lpq
LEX_H= medconflex.h ddlex.h odlex.h
LEX_C= lex.medconf.c lex.ddprep.c lex.odfeed.c
OBJECTS= main.o alert_sql.o cuda_f.o dd.o od.o med_config.o lex.medconf.o lex.ddprep.o lex.odfeed.o
DEBUG= -g

all: med

med: ${OBJECTS}
	${CC} -o med $^ -I${INC_D} ${LIBS}


# object files
lex.medconf.o: lex.medconf.c medconflex.h
	${CC} -c ${SRC_D}lex.medconf.c -I${INC_D}

lex.ddprep.o: lex.ddprep.c ddlex.h
	${CC} -c ${SRC_D}lex.ddprep.c -I${INC_D}

lex.odfeed.o: lex.odfeed.c odlex.h
	${CC} -c ${SRC_D}lex.odfeed.c -I${INC_D}

med_config.o: medconflex.h
	${CC} -c ${SRC_D}med_config.c -I${INC_D}

od.o:
	${CC} -c ${SRC_D}od.c -I${INC_D}

dd.o:
	${CC} -c ${SRC_D}dd.c -I${INC_D}

cuda_f.o:
	${NVCC} -c -arch=sm_20 ${SRC_D}cuda_f.cu

alert_sql.o:
	${CC} -c ${SRC_D}alert_sql.c -I${INC_D} -I/usr/include/postgresql

main.o: medconflex.h ddlex.h odlex.h
	${CC} -c ${SRC_D}main.c -I${INC_D}


# c file output from flex
lex.medconf.c medconflex.h:
	${LEX} -o ${SRC_D}lex.medconf.c ${LEX_D}configlex.l

lex.ddprep.c ddlex.h:
	${LEX} -o ${SRC_D}lex.ddprep.c ${LEX_D}preplex.l

lex.odfeed.c odlex.h:
	${LEX} -o ${SRC_D}lex.odfeed.c ${LEX_D}feedlex.l


.PHONY: clean
clean:
	rm med *.o ${INC_D}*lex.h ${SRC_D}lex.*
