#ifndef OD_H
#define OD_H

#define ODSTAT_TOK 0
#define ODSTAT_ID 1

void  od_init();
void  od_free();
void  od_setid(char *);
void  od_addtoken(char *);
int  *od_getarray();
int   od_getlen();
char *od_getid();

#endif
