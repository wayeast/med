#ifndef MED_CONFIG_H
#define MED_CONFIG_H

void  medconf_setindbname(const char *);
void  medconf_setinhostaddr(const char *);
void  medconf_setinport(const char *);
void  medconf_setinuser(const char *);
void  medconf_setinconnecttimeout(const char *);
void  medconf_setinpassword(const char *);
void  medconf_setintable(const char *);
void  medconf_setinputcol(const char *);
void  medconf_setinputseparator(const char *);

void  medconf_setoutdbname(const char *);
void  medconf_setouthostaddr(const char *);
void  medconf_setoutport(const char *);
void  medconf_setoutuser(const char *);
void  medconf_setoutconnecttimeout(const char *);
void  medconf_setoutpassword(const char *);
void  medconf_setouttable(const char *);
void  medconf_setoutorigcol(const char *);
void  medconf_setoutdestcol(const char *);
void  medconf_setoutclobber(const char *);

void  medconf_setzerorangeevals(const char *);
void  medconf_setlinearrangeevals(const char *);
void  medconf_setdiscreteevals(const char *);

char *medconf_getindbname(void);
char *medconf_getinhostaddr(void);
char *medconf_getinport(void);
char *medconf_getinuser(void);
char *medconf_getinconnecttimeout(void);
char *medconf_getinpassword(void);
char *medconf_getintable(void);
char *medconf_getinputcol(void);
char *medconf_getinputseparator(void);

char *medconf_getoutdbname(void);
char *medconf_getouthostaddr(void);
char *medconf_getoutport(void);
char *medconf_getoutuser(void);
char *medconf_getoutconnecttimeout(void);
char *medconf_getoutpassword(void);
char *medconf_getouttable(void);
char *medconf_getoutorigcol(void);
char *medconf_getoutdestcol(void);
int   medconf_isclobber(void);

int  *medconf_getetab(void);
int   medconf_getnoevals(void);

void  medconf_cleanup(void);
void  medconf_configure(void);

#endif
