package cs475.NeuralNetwork;

/******************************************************************************
 *  Compilation:  javac Matrix.java
 *  Execution:    java Matrix
 *
 *  A bare-bones collection of static methods for manipulating
 *  matrices.
 *
 ******************************************************************************/

public class Matrix {

    // return a random m-by-n matrix with values between 0 and 1
    public static double[][] random(int m, int n) {
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = Math.random();
        return C;
    }

    // return n-by-n identity matrix I
    public static double[][] identity(int n) {
        double[][] I = new double[n][n];
        for (int i = 0; i < n; i++)
            I[i][i] = 1;
        return I;
    }

    // return a mxn matrix with all elements are 1
    public static double[][] allOnes(int m, int n){
        double[][] A = new double[m][n];
        for (int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                A[i][j] = 1.0;
            }
        }
        return A;
    }

    // return x^T y
    public static double dot(double[] x, double[] y) {
        if (x.length != y.length)
            throw new RuntimeException("Illegal vector dimensions.");

        double sum = 0.0;
        for (int i = 0; i < x.length; i++)
            sum += x[i] * y[i];
        return sum;
    }

    // Perform element wise multiplication
    public static double[][] dotMultiply(double[][] x, double[][] y){
        if (x.length != y.length || x[0].length != y[0].length) {
            System.out.printf("[%d x %d] [%d x %d]\n", x.length, x[0].length, y.length, y[0].length);
            throw new RuntimeException("Illegal vector dimensions.");
        }

        double[][] C = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++){
            for(int j = 0; j < x[0].length; j++){
                C[i][j] = x[i][j] * y[i][j];
            }
        }
        return C;
    }

    // return C = A^T
    public static double[][] transpose(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[n][m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[j][i] = A[i][j];
        return C;
    }

    // return C = A + B
    public static double[][] add(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = A[i][j] + B[i][j];
        return C;
    }

    // return C = A - B
    public static double[][] subtract(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = A[i][j] - B[i][j];
        return C;
    }

    // substract a given scalar from each element in the matrix
    public static double[][] substract(double[][] A, double b){
        for (int i = 0; i < A.length; i++){
            for (int j = 0; j < A[0].length; j++){
                A[i][j] -= b;
            }
        }
        return A;
    }

    // return C = A * B
    public static double[][] multiply(double[][] A, double[][] B) {
        int mA = A.length;
        int nA = A[0].length;
        int mB = B.length;
        int nB = B[0].length;
        if (nA != mB)
            System.out.printf("[%d x %d] [%d x %d]\n", mA, nA, mB, nB);
        if (nA != mB) throw new RuntimeException("Illegal matrix dimensions");
        double[][] C = new double[mA][nB];
        for (int i = 0; i < mA; i++)
            for (int j = 0; j < nB; j++)
                for (int k = 0; k < nA; k++)
                    C[i][j] += A[i][k] * B[k][j];
        return C;
    }

    // matrix-vector multiplication (y = A * x)
    public static double[] multiply(double[][] A, double[] x) {
        int m = A.length;
        int n = A[0].length;
        if (x.length != n) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                y[i] += A[i][j] * x[j];
        return y;
    }


    // vector-matrix multiplication (y = x^T A)
    public static double[] multiply(double[] x, double[][] A) {
        int m = A.length;
        int n = A[0].length;
        if (x.length != m) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[n];
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                y[j] += A[i][j] * x[i];
        return y;
    }

    // multiply elements by a given scalar.
    public static double[][] multiply(double[][] x, double a){
        for (int i = 0; i < x.length; i++){
            for (int j = 0; j < x[0].length; j++){
                x[i][j] *= a;
            }
        }
        return x;
    }

    // return the total sum of all the elements in the given matrix.
    public static double sum(double[][] A){
        double sum = 0.0;
        for (int i = 0; i < A.length; i++){
            for (int j = 0; j < A[0].length; j++){
                sum += A[i][j];
            }
        }
        return sum;
    }

    // return the total sum of all the elements in the given matrix.
    public static double sumExceptFirst(double[][] A){
        double sum = 0.0;
        for (int i = 0; i < A.length; i++){
            for (int j = 0; j < A[0].length; j++){
                if (j == 0)
                    continue;
                sum += A[i][j];
            }
        }
        return sum;
    }

    // return a matrix with every element is the Math.pow(org, pow) of the given matrix.
    public static double[][] powOfElement(double[][] A, int pow){
        double[][] B = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++){
            for (int j = 0; j < A[0].length; j++){
                B[i][j] = Math.pow(A[i][j], pow);
            }
        }
        return B;
    }


//    // test client
//    public static void main(String[] args) {
//        StdOut.println("D");
//        StdOut.println("--------------------");
//        double[][] d = { { 1, 2, 3 }, { 4, 5, 6 }, { 9, 1, 3} };
//        StdArrayIO.print(d);
//        StdOut.println();
//
//        StdOut.println("I");
//        StdOut.println("--------------------");
//        double[][] c = Matrix.identity(5);
//        StdArrayIO.print(c);
//        StdOut.println();
//
//        StdOut.println("A");
//        StdOut.println("--------------------");
//        double[][] a = Matrix.random(5, 5);
//        StdArrayIO.print(a);
//        StdOut.println();
//
//        StdOut.println("A^T");
//        StdOut.println("--------------------");
//        double[][] b = Matrix.transpose(a);
//        StdArrayIO.print(b);
//        StdOut.println();
//
//        StdOut.println("A + A^T");
//        StdOut.println("--------------------");
//        double[][] e = Matrix.add(a, b);
//        StdArrayIO.print(e);
//        StdOut.println();
//
//        StdOut.println("A * A^T");
//        StdOut.println("--------------------");
//        double[][] f = Matrix.multiply(a, b);
//        StdArrayIO.print(f);
//        StdOut.println();
//    }
}

