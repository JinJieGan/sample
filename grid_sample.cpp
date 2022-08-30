

//#include <grid_sample.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <chrono>
using namespace std;

static inline bool within_bounds_2d(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}


class Timer {
public:
  Timer() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

//using namespace cl::sycl;

vector<vector<vector<vector<float> > > > grid_simple(vector<vector<vector<vector<float> > > > input,
    vector<vector<vector<vector<float> > > > grid,
    vector<vector<vector<vector<float> > > > output) {

  int N = input.size();
  int C = input[0].size();
  int IH = input[0][0].size();
  int IW = input[0][0][0].size();
  int H = grid[0].size();
  int W = grid[0][0].size();
  float zero_point = 0.;
  

  // loop over each output pixel
  int n, h, w, c;
//#pragma omp parallel for private(n, h, w, c)
  for (n = 0; n < N; ++n) {
    for (h = 0; h < H; ++h) {
      for (w = 0; w < W; ++w) {
        // get the corresponding input x, y co-ordinates from grid
        
        float rx = grid[n][h][w][0];
        float ry = grid[n][h][w][1];
        //cout << "real x:" << rx << ", real y: "<< ry <<endl;

        // normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
        float ix = ((rx + 1) / 2) * (IW-1);
        float iy = ((ry + 1) / 2) * (IH-1);

        // get NE, NW, SE, SW pixel values from (x, y)
        int ix_nw = std::floor(ix);
        int iy_nw = std::floor(iy);
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;
        //cout << "ix_nw: " << ix_nw << ", iy_nw: " << iy_nw << ", ix: " << ix << ", iy: "<< iy <<endl;
        // get surfaces to each neighbor:
        float nw = (ix_se - ix)    * (iy_se - iy);
        float ne = (ix    - ix_sw) * (iy_sw - iy);
        float sw = (ix_ne - ix)    * (iy    - iy_ne);
        float se = (ix    - ix_nw) * (iy    - iy_nw);
        //cout << "nw: " << nw << ", ne: " << ne << ", sw: " << sw << ", se: "<< se <<endl;

        // calculate bilinear weighted pixel value and set output pixel
        for (c = 0; c < C; ++c) {
          //   (c, iy_nw, ix_nw) * nw + (c, iy_ne, ix_ne) * ne
          // + (c, iy_sw, ix_sw) * sw + (c, iy_se, ix_se) * se
          float res = 0;
            res += within_bounds_2d(iy_nw, ix_nw, IH, IW)
                ? input[n][c][iy_nw][ix_nw] * nw
                : zero_point * nw;
            //cout <<"input data: "<< zero_point * nw<< ", res1:" << res << endl;
            res += within_bounds_2d(iy_ne, ix_ne, IH, IW)
                ? input[n][c][iy_ne][ix_ne] * ne
                : zero_point * ne;
            //cout <<"input data: "<<zero_point * ne<< ", res2:" << res << endl;
            res += within_bounds_2d(iy_sw, ix_sw, IH, IW)
                ? input[n][c][iy_sw][ix_sw] * sw
                : zero_point * sw;
            //cout <<"input data: "<<zero_point * sw<< ", res3:" << res << endl;
            res += within_bounds_2d(iy_se, ix_se, IH, IW)
                ? input[n][c][iy_se][ix_se] * se
                : zero_point * se;
            //cout <<"input data: "<<zero_point * se << ", res4:" << res << endl;
            output[n][c][h][w] = res; // std::round(res);
            //cout << "output: "<< output[n][c][h][w] <<endl;
        }
      }
    }
  }
  return output;
}

int main(){
    cout<<"begin"<<endl;
    int in_N = 1;
    int in_C = 1;
    int in_H = 512;
    int in_W = 960;
    int out_H = 1024;
    int out_W = 960;
    vector<vector<vector<vector<float> > > > input(in_N,vector<vector<vector<float> > >(in_C, vector<vector<float> >(in_H, vector<float>(in_W,1.0)))); 
    vector<vector<vector<vector<float> > > > grid(in_N,vector<vector<vector<float> > >(out_H, vector<vector<float> >(out_W, vector<float>(2,0.0000))));
    vector<vector<vector<vector<float> > > > output(in_N,vector<vector<vector<float> > >(in_C, vector<vector<float> >(out_H, vector<float>(out_W,0.0))));
    /*
    vector<float> list_grid = {-1.0000, -1.0000,
          -1.0000, -0.5000,
          -1.0000,  0.0000,
          -1.0000,  0.5000,
          -1.0000,  1.0000,

         -0.5000, -1.0000,
          -0.5000, -0.5000,
          -0.5000,  0.0000,
          -0.5000,  0.5000,
          -0.5000,  1.0000,

           0.0000, -1.0000,
           0.0000, -0.5000,
           0.0000,  0.0000,
           0.0000,  0.5000,
           0.0000,  1.0000,

          0.5000, -1.0000,
           0.5000, -0.5000,
           0.5000,  0.0000,
           0.5000,  0.5000,
           0.5000,  1.0000,

          1.0000, -1.0000,
           1.0000, -0.5000,
           1.0000,  0.0000,
           1.0000,  0.5000,
           1.0000,  1.0000};
    int idx = 0;
    for (int i= 0; i < in_N; ++i){
        for(int j = 0; j < out_H; ++j){
            for(int k = 0; k < out_W; ++k){
                    grid[i][j][k][0] = list_grid[idx++];
                    grid[i][j][k][1] = list_grid[idx++];
                    //cout<<"grid x： "<< grid[i][j][k][0] << " , grid y: "<< grid[i][j][k][1]<<endl;
            }
        }
    }
    float pixel = 1.0;
    for (int i= 0; i < in_N; ++i){
        for(int j = 0; j < in_C; ++j){
            for(int k = 0; k < in_H; ++k){
                for (int l = 0; l < in_W; ++l){
                    input[i][j][k][l] = pixel++ ;
                    //cout << "input : " << input[i][j][k][l] <<endl;
                }    
            }
        }
    }
    */
    Timer timer;
    auto start = timer.Elapsed(); 
    output = grid_simple(input,grid,output);
    auto end = timer.Elapsed();
    float total_time = (end - start) * 1000.f;    
    std::cerr << "sample time=" << total_time << " msec\n";
}
