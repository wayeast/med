#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "od.h"
#include "dd.h"

#define ARR_INIT 25
#define ID_INIT  21

struct {
    int  *array;
    int   len;
    int   a_cap;
    char *id;
} od;

int  *od_getarray() { return od.array; }
int   od_getlen()   { return od.len; }
char *od_getid()    { return od.id; }



void od_free() {
    free(od.array);
    free(od.id);
}

void od_init() {
    if (! (od.array = (int *) malloc(ARR_INIT * sizeof(int)) ))
        fprintf(stderr, "ERROR: unable to allocate memory for od.array\n");
    if (! (od.id = (char *) malloc(ID_INIT * sizeof(char)) ))
        fprintf(stderr, "ERROR: unable to allocate memory for od.id\n");
    od.len = 0;
    od.a_cap = ARR_INIT;
}

void od_setid(char *id){
    if ( strlen(id) > (ID_INIT - 1) ) {
        fprintf(stderr, "ERROR: unable to set id; '%s' too long.\n", id);
        exit(0);
    }
    strcpy(od.id, id);
}

void od_addtoken(char *tok) {
    int t;
    t = dd_lookup(tok);
    if (od.len == od.a_cap) {
        int *new_array;
        if (! (new_array = (int *) malloc( (od.a_cap + ARR_INIT) * sizeof(int) ))) {
            fprintf(stderr, "ERROR: unable to allocate memory for expanded od.array.\n");
            od_free();
            exit(0);
        }
        int i;
        for (i=0; i<od.len; i++) new_array[i] = od.array[i];
        free(od.array);
        od.array = new_array;
        od.a_cap += ARR_INIT;
    }
    od.array[od.len] = t;
    od.len++;
}
