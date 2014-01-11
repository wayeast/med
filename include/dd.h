#ifndef DD_H
#define DD_H

void   dd_init();
void   dd_freetable();
void   dd_break();
int    dd_lookup(char *);
void   dd_addentry(char *);
void   dd_addtarg(char *);

char **dd_getvtab();
int    dd_getventries();
int   *dd_gettargs();
int   *dd_gettbi();
char **dd_gettids();
int    dd_getntargs();
int    dd_getstargs();


#endif
