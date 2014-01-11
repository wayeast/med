/*
 * A try at implementing the MED algorithm on a GPU.
 *
 * ===  med-kernel:  ===
 * Essentially an implementation of the MED algorithm from Jurafsky
 * and Martin tailored to a GPU.
 * Each string to be compared is first converted into an int array, 
 * where each element is the index into the vocabulary _dd_ table that
 * was built beforehand.
 * Before the kernel is called, the device is prepared with a very large
 * int array representing all target strings (all strings in the alert_data
 * table) laid out end to end.  A few other small pieces of information that
 * remain constant are also sent:  ntargets and tbi.  On the device is
 * also reserved space for an edits array where each block will store its
 * output at the end of every cycle.
 * Before each kernel invocation, a new orig array _od_ (also an int array
 * representing indices of tokens into the vocabulary table) and an olen int
 * are reserved and CudaMemCopied to the device.  Also, space for each block to 
 * make its distance calculations is reserved.
 * _distance_ is an array with [ntargets * (olen + 1) * 2] elements.  Each half of
 * the array is treated as one column of the J&M distance matrix (see below).
 * Where the original MED algorithm uses a row++ loop inside a column++ loop,
 * this algorithm swaps between 'columns'.  I can do this because I'm only
 * concerned with the edit distance -- not the moves required to change the
 * string.
 * The number of rows each thread handles is the length of the original string,
 * thus each iteration of the inner loop will take the same number of steps for
 * each thread.  The number of columns each thread handles, however, is a
 * function of the length of the target string, which will vary from one thread
 * to the next.  Hence, some threads will make more or fewer outer loop
 * iterations with respect to other threads.
 * Conceptually, think of the relationship between threads and loops as such:
 *
 * |     |     |     |       |  |
 * |     |     |     |       |  |
 * |     |     |     |       |  |
 * |     |     |     |       |  |
 * |     |     |     |       |  |
 * ====  ==    ===== ======= == ====
 * tid=0 tid=1 tid=2 tid=3 ...
 *
 * In fact, the number of rows in _distance_ is olen * ntargets == olen *
 * nblocks.  In this diagram, think of each vertical line as being an original string
 * that will be used by one thread.  Each vertical line is in fact a segment of the
 * array named _distance_.
 * The x axis is truer to reality - this is the targets array of all alert
 * strings laid end-to-end.
 * Each delimited cartesian space is the responsibility of one thread.
 * Each vertical line is an int array representing the original string, so there are the
 * same number of rows across the board.  Since the length of each target
 * string is variable, the space calculated by one thread is not constant.
 * 
 * I've designed the 2-column distance table to be a linear array representing
 * 2 columns.  This is because of the greater simplicity of allocating the necessary
 * memory on the device and of accessing its elements.  To accomodate the new
 * circumstances, I've added two pointers cur_col and last_col which will point to
 * either the head of distance (the distance pointer itself) or to the address halfway
 * into the distance array, depending on which column of the distance table is being
 * filled versus which was just filled.  I have also added macros current and last
 * to return the correct pointer as a function of cindex % 2.
 */

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

// The next two kernel definitions are variations on the same theme.  They accomplish the
// same task -- one with a while loop, the other with a for loop.  Duplication strictly for
// experimentation purposes to test performance.
__global__ void med_kernel_while( int *orig,      // original string; must be freed at kernel termination
                            int olen,      // length of orig
                            int *edits,     // result array where each block stores its calculation
                            int *targets,   // compound array of all other target strings
                            int *distance, // scratch pad for blocks to build MED table; must be
                                            // freed at kernel termination
                            int *tbi,        // (target-begin-index) array of indices where each block
                                            // will find the start of its
                                            // target string
                            int stargets,    // total number of elements in targets array
                            int ntargets
                            ) {
    int tid = threadIdx.x;
    int tlength = tbi[tid+1] - tbi[tid];
    int *target = &targets[tbi[tid]];
    int *dcol1 = &distance[(olen+1)*tid];
    int *dcol2 = &distance[(olen+1)*tid + (olen+1)*ntargets];
    int *current = dcol1;
    int *last = dcol2;
    int row, col, n1, n2, n3, petitmin, grandmin;
    // initialize first column
    row = col = 0;
    current[row] = 0;
    for (row=1; row<olen+1; row++) current[row] = current[row-1] + 1;
    // fill in distance matrix
    while ( col<tlength ) {
        current = (current == dcol1 ? dcol2 : dcol1);
        last    = (last    == dcol1 ? dcol2 : dcol1);
        // bottom row must be initialized piecemeal
        row = 0;
        current[row] = last[row] + 1;
        for (row=1; row<olen+1; row++) {
            // calculate three possible values...
            n1 = last[row] + 1;
            n2 = current[row-1] + 1;
            n3 = (orig[row-1] == target[col] ? last[row-1] : last[row-1] + 2);
            // ...and find minimum
            petitmin = (n1 < n2 ? n1 : n2);
            grandmin = (petitmin < n3 ? petitmin : n3);
            // add row
            current[row] = grandmin;
        }  // end inner loop = build rows of a column
        col++;
    }  // end outer loop = build columns of target

    // wait for all threads to finish and submit MED
    edits[tid] = current[olen];
    __syncthreads();

}  // end med_kernel
 
__global__ void med_kernel_for( int *orig,      // original string; must be freed at kernel termination
                            int olen,      // length of orig
                            int *edits,     // result array where each block stores its calculation
                            int *targets,   // compound array of all other target strings
                            int *distance, // scratch pad for blocks to build MED table; must be
                                            // freed at kernel termination
                            int *tbi,        // (target-begin-index) array of indices where each block
                                            // will find the start of its
                                            // target string
                            int stargets,    // total number of elements in targets array
                            int ntargets
                            ) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (! (tid < ntargets)) return;
    int tlength = tbi[tid+1] - tbi[tid];
    int *target = &targets[tbi[tid]];
    int *dcol1 = &distance[(olen+1)*tid];
    int *dcol2 = &distance[(olen+1)*tid + (olen+1)*ntargets];
    int *current = dcol1;
    int *last = dcol2;
    int row, col, n1, n2, n3, petitmin, grandmin;
    // initialize first column
    row = col = 0;
    current[row] = 0;
    for (row=1; row<olen+1; row++) current[row] = current[row-1] + 1;
    // fill in distance matrix
    for (col=0; col<tlength; col++) {
        current = (current == dcol1 ? dcol2 : dcol1);
        last    = (last    == dcol1 ? dcol2 : dcol1);
        // bottom row must be initialized piecemeal
        row = 0;
        current[row] = last[row] + 1;
        for (row=1; row<olen+1; row++) {
            // calculate three possible values...
            n1 = last[row] + 1;
            n2 = current[row-1] + 1;
            n3 = (orig[row-1] == target[col] ? last[row-1] : last[row-1] + 2);
            // ...and find minimum
            petitmin = (n1 < n2 ? n1 : n2);
            grandmin = (petitmin < n3 ? petitmin : n3);
            // add row
            current[row] = grandmin;
        }  // end inner loop = build rows of a column
    }  // end outer loop = build columns of target

    // wait for all threads to finish and submit MED
    __syncthreads();
    edits[tid] = current[olen];
    //edits[tid] = grandmin;

}  // end med_kernel
 
extern "C"
void call_med_kernel(
        int *orig,      // original string; must be freed at kernel termination
        int olen,      // length of orig
        int *edits,     // result array where each block stores its calculation
        int *targets,   // compound array of all other target strings
        int *distance, // scratch pad for blocks to build MED table; must be
        int *tbi,
        int starg,
        int ntarg
        ) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxthr, blocks;
    maxthr = prop.maxThreadsPerBlock;
    blocks = ntarg / maxthr + 1;

    med_kernel_for<<<blocks, maxthr>>>( orig, olen, edits, targets, distance, tbi, starg, ntarg );
}


extern "C"
void call_cudaMalloc(int **d_ptr, int bytes) {
    cudaMalloc( (void**) d_ptr, bytes );
}

extern "C"
void call_cudaMemcpy(int *to_ptr, int *from_ptr, int bytes, int flag){
    cudaMemcpy(to_ptr, from_ptr, bytes, (flag ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice));
}

extern "C"
void call_cudaFree(int *dev_ptr) {
    cudaFree(dev_ptr);
}
