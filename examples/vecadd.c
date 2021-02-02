__global__ void add(float* z, float* x, float* y, int N){
    // program id
    int pid = get_program_id(0);
    // create arrays of pointers
    int offset[1024] = pid * 1024 + 0 ... 1024;
    float* pz[1024] = z + offset;
    float* px[1024] = x + offset;
    float* py[1024] = y + offset;
    // bounds checking
    bool check[1024] = offset < N;
    // write-back
    *?(check)pz = *?(check)px + *?(check)py;
}