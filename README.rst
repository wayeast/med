Repo:  med
----------------
Features
----------
- *C*
- *CUDA*
- *Lex*
- *PostgreSQL*

Description
-------------
This code implements the common Minimum Edit Distance algorithm on a GPU,
reducing the time required to process a large text library from
approximately 1 week with a single-processor CPU to less than 20 minutes.
It was written as part of a research grant to create a tool that could 
help locate entries in an electronic technical manual that were redundant
and target them for removal.  The difficulty was that many entries were
indeed redundant even though they differed only slightly in their wording
or punctuation.  Thus, a simple pattern-matching solution was not
sufficient to track them down.

Examples of targeted entries include

1.
    a. ``To prevent pinching or chafing of wires and cabling, check the positioning of the wires and cabling while closing the panel.``
    b. ``To prevent pinching or chafing of the wires and cabling, check the positioning of the wires and cabling while installing the panel.``
2.
    a. ``A continuous extrusion of sealant is required around head of screw.``
    b. ``A continuous extrusion of sealant is required around hole of screw.``
3.
    a. ``Overtorquing threads coated with corrosion preventive compound...``
    b. ``Overtorquing of threads coated with corrosion preventive compound...``

Since many of the differences between entries really only amount to slight
modifications of a word or two, I looked to the Minimum Edit Distance (MED)
algorithm as a solution.  The MED algorithm is one that has been used in 
many areas to solve problems that can be reduced to calculating the number
of changes required to move from one sequence to another.  Spell-checkers, 
for example, will often use a version of MED to suggest possible 
corrections to unknown words they encounter by counting the minimum number
of letter changes needed to get to a known word.  This implementation of
the algorithm does not count changes on the level of individual letters but
rather on the level of whole words.

The dataset I was working with was a library of nearly 20,000 entries.  To
calculate the distance between two sequences, the MED algorithm constructs
a matrix whose size is the length of the origin sequence by the length of
the destination sequence.  Thus to find the edit distance from an 8-element
sequence to a 9-element sequence requires building an 8x9 (actually, 9x10
to be exact) matrix and filling in each square of the matrix one by one. 
For working with large data sets, this can quickly become a rather costly
procedure in terms of time required.  My initial implementation of the MED
algorithm, written in Python, running on a 64-bit, single-processor desktop
Linux machine, took approximately 1 week to calculate the distance from each
entry in the library to every other entry -- clearly much more of a time 
investment than desirable.

The code here implements the same algorithm in CUDA for execution on an
Nvidia Tesla GPU and hence reduces the time required for processing the entire
library to under 20 minutes.  The strategy is as follows: since this 
implementation is comparing words, each word is assigned an integer value
(punctuation is ignored) and each entry in the library is represented as a
sequence of integers (**source/dd.c** handles the construction and accessing 
of a vocabulary table that allows for this transformation); the entire
library is represented as a single, giant integer array and moved en masse
to the GPU; one at a time, each entry is then converted into an integer 
array (**source/od.c** handles the conversion of individual entries) and moved
to the GPU; as many CUDA threads are created as there are entries in the 
entire library, and each thread executes the MED algorithm comparing the 
individual sequence against the library entry for which it is responsible 
(**source/cuda\_f.cu** contains wrapper functions for copying data to and from
the GPU, as well as the main kernel that performs the MED algorithm); an edit
distance array is created, representing the edit distances between the single
entry and every other entry in the library; this array is copied back to the
CPU, which culls the values of interest and returns the results.
**source/alert\_sql.c** contains the code that allows the process to interact
with a PostgreSQL database that holds the library being processed.  All
runtime parameters are set from a configuration file that is read at the
start of any invocation of the program.  **config.rc** is the file I used
for running the program from my computer.
