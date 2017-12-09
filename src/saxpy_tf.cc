// tensorflow/cc/example/example.cc

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include <chrono>

using namespace tensorflow;
using namespace tensorflow::ops;

class saxpy_timer {
public:
   saxpy_timer() {
      reset();
   }
   void reset() {
      t0_ = std::chrono::high_resolution_clock::now();
   }
   double elapsed(bool reset_timer = false) {
      std::chrono::high_resolution_clock::time_point t =
            std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> time_span = std::chrono::duration_cast<
            std::chrono::duration<double>>(t - t0_);
      if (reset_timer)
         reset();
      return time_span.count();
   }
   double elapsed_msec(bool reset_timer = false) {
      return elapsed(reset_timer) * 1000;
   }
private:
   std::chrono::high_resolution_clock::time_point t0_;
};

static void show_output(const char *op, const std::vector<Tensor> &outs) {
   LOG(INFO)<< "*********************** " << op << " **************************";
   LOG(INFO) << "output size: " << outs.size();
   for (auto el : outs)
   LOG(INFO) << el.DebugString();
   LOG(INFO) << "";
}

int main() {
   LOG(INFO)<< "Begin";
   Scope root = Scope::NewRootScope();

   //const int N = 16;
   const unsigned N = (1 << 26);
   const float XVAL = 3.0f;
   const float YVAL = 2.0f;
   const float AVAL = 1.5f;

   ClientSession session(root);
   std::vector<Tensor> outs;

   LOG(INFO)<< "Start";

   //
   // Init Y
   //
   auto y = Variable(root, { N }, DT_FLOAT);
   auto inity = Assign(root, y, Const(root, YVAL, { N }));
   TF_CHECK_OK(session.Run( { inity }, &outs));
   show_output("init y", outs);

   //
   // Init X
   //
   auto x = Variable(root, { N }, DT_FLOAT);
   auto initx = Assign(root, x, Const(root, XVAL, { N }));
   TF_CHECK_OK(session.Run( { initx }, &outs));
   show_output("init x", outs);

   //
   // Init A
   //
#if 1
   auto a = Const<float>(root, AVAL);
#else
   auto a = Variable(root, { N }, DT_FLOAT);
   auto inita = Assign(root, a, Const(root, AVAL, { N }));
   TF_CHECK_OK(session.Run( { inita }, &outs));
   show_output("init a", outs);
#endif

   //
   // Saxpy
   //

   auto mul = Multiply(root, x, a);
   auto saxpy = AssignAdd(root, y, mul);

   auto &op = saxpy;

   saxpy_timer t;
   //TF_CHECK_OK( session.Run({saxpy}, &outs) );
   session.Run( { op }, nullptr);
   double elapsed = t.elapsed_msec();

   show_output("saxpy", outs);
   LOG(INFO)<< "Elapsed: " << elapsed << " ms";

   //
   // Check errors
   //
   auto diff = Subtract(root, y, Const<float>(root, YVAL + AVAL * XVAL));
   auto error = ReduceSum(root, diff, 0);
   TF_CHECK_OK(session.Run( { error }, &outs));
   show_output("error", outs);

   LOG(INFO) << "Success";
   return 0;
}

/*
BUILD:
---------
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "example",
    srcs = ["example.cc"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
    ],
)

$ bazel build -c opt --copt=-march=native --copt=-mfpmath=both --config=opt --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 --config=cuda --copt=-Wno-sign-compare --copt=-Wno-unused-variable //tensorflow/cc/example:example
$ bazel run -c opt --copt=-march=native --copt=-mfpmath=both --config=opt --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 --config=cuda --copt=-Wno-sign-compare --copt=-Wno-unused-variable //tensorflow/cc/example:example
*/
