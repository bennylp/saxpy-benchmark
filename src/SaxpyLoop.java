public class SaxpyLoop {
   public static void main(String []args) {
	   int N = 1 << 26;
	   float XVAL = (float)(10 * Math.random());
	   float YVAL = (float)(10 * Math.random());
	   float AVAL = (float)(10 * Math.random());
	   float[] x = new float[N];
	   float[] y = new float[N];

	   System.out.println("N: " + N);
	   
	   for (int i=0; i<N; i++) {
		   x[i] = XVAL;
		   y[i] = YVAL;
	   }
	   
	   double startTime = System.nanoTime();
	   for (int i=0; i<N; i++) {
		   y[i] += AVAL * x[i];
	   }
	   double endTime = System.nanoTime();
	   System.out.println("Elapsed: " + (endTime - startTime) / 1000000.0 + " ms");
	   
	   float error = 0;
	   for (int i=0; i<N; i++) {
		   error += Math.abs(y[i] - (YVAL + AVAL * XVAL));
	   }
	   
	   System.out.println("Error: " + error);
   }
}
