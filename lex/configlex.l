/* Lexer to read in configuration file */

%{
#include "med_config.h"
%}

comment     #.*\n
inputdb     ^Input_db
outputdb    ^Output_db
whitesp     [ \t]+
blankline   ^[ \t]*\n

name        [A-Za-z][A-Za-z0-9_\.\-]*
number      [0-9]+
addr        ([0-9]+\.){3}[0-9]+

%x INPUT_S
%X OUTPUT_S
%option noyywrap
%option prefix="medconf"
%option header-file="include/medconflex.h"

%%

{inputdb}                               { BEGIN INPUT_S; }
<INPUT_S>{blankline}                    { BEGIN 0; }
<INPUT_S>{outputdb}                     { BEGIN OUTPUT_S; }
<INPUT_S>{whitesp}                      ;
<INPUT_S>\n                             ;
<INPUT_S>{comment}                  ;
<INPUT_S>dbname={name}              { medconftext += 7;  medconf_setindbname(medconftext); }
<INPUT_S>hostaddr={addr}            { medconftext += 9;  medconf_setinhostaddr(medconftext); }
<INPUT_S>port={number}              { medconftext += 5;  medconf_setinport(medconftext); }
<INPUT_S>user={name}                { medconftext += 5;  medconf_setinuser(medconftext); }
<INPUT_S>connect_timeout={number}   { medconftext += 16; medconf_setinconnecttimeout(medconftext); }
<INPUT_S>password={name}            { medconftext += 9;  medconf_setinpassword(medconftext); }
<INPUT_S>table={name}               { medconftext += 6;  medconf_setintable(medconftext); }
<INPUT_S>input_col={name}           { medconftext += 10; medconf_setinputcol(medconftext); }
<INPUT_S>input_separator={name}     { medconftext += 16; medconf_setinputseparator(medconftext); }

{outputdb}                              { BEGIN OUTPUT_S; }
<OUTPUT_S>{blankline}                   { BEGIN 0; }
<OUTPUT_S>{inputdb}                     { BEGIN INPUT_S; }
<OUTPUT_S>{whitesp}                     ;
<OUTPUT_S>\n                            ;
<OUTPUT_S>{comment}                 ;
<OUTPUT_S>dbname={name}             { medconftext += 7; medconf_setoutdbname(medconftext); }
<OUTPUT_S>hostaddr={addr}           { medconftext += 9; medconf_setouthostaddr(medconftext); }
<OUTPUT_S>port={number}             { medconftext += 5; medconf_setoutport(medconftext); }
<OUTPUT_S>user={name}               { medconftext += 5; medconf_setoutuser(medconftext); }
<OUTPUT_S>connect_timeout={number}  { medconftext += 16; medconf_setoutconnecttimeout(medconftext); }
<OUTPUT_S>password={name}           { medconftext += 9; medconf_setoutpassword(medconftext); }
<OUTPUT_S>table={name}              { medconftext += 6; medconf_setouttable(medconftext); }
<OUTPUT_S>out_origcol={name}          { medconftext += 12; medconf_setoutorigcol(medconftext); }
<OUTPUT_S>out_destcol={name}    { medconftext += 12; medconf_setoutdestcol(medconftext); }
<OUTPUT_S>clobber={name}            { medconftext += 8; medconf_setoutclobber(medconftext); }

edit_range={number}                     { medconftext += 11; medconf_setzerorangeevals(medconftext); }
edit_range={number}-{number}            { medconftext += 11; medconf_setlinearrangeevals(medconftext); }
edit_range=({number})(,{number})+       { medconftext += 11; medconf_setdiscreteevals(medconftext); }

{blankline}                             ;
{comment}                               ;
{whitesp}                               ;

%%


