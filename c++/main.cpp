#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std;

void flatten(Matrix<double, 4, 4> &g, Matrix<double,1,12> &y) {
    y(0) = g(0,0);
    y(1) = g(0,1);
    y(2) = g(0,2);
    y(3) = g(1,0);
    y(4) = g(1,1);
    y(5) = g(1,2);
    y(6) = g(2,0);
    y(7) = g(2,1);
    y(8) = g(2,2);
    y(9) = g(0,3);
    y(10) = g(1,3);
    y(11) = g(2,3);
}

void unflatten(Matrix<double, 1, 12> &y, Matrix<double, 4, 4> &g) {
    g(0,0) = y(0);
    g(0,1) = y(1);
    g(0,2) = y(2);
    g(1,0) = y(3);
    g(1,1) = y(4);
    g(1,2) = y(5);
    g(2,0) = y(6);
    g(2,1) = y(7);
    g(2,2) = y(8);
    g(0,3) = y(9);
    g(1,3) = y(10);
    g(2,3) = y(11);
    g(3,0) = 0;
    g(3,1) = 0;
    g(3,2) = 0;
    g(3,3) = 1;
}

void skew(Matrix<double, 3, 1> &x, Matrix<double, 3, 3> &y) {
    y(0,1) = -x(2);
    y(0,2) = x(1);
    y(1,0) = x(2);
    y(1,2) = -x(0);
    y(2,0) = -x(1);
    y(2,1) = x(0);
    y(0,0) = 0;
    y(1,1) = 0;
    y(2,2) = 0;
}

void adjoint(Matrix<double, 6, 1> &x, Matrix<double, 6, 6> &y) {
    Vector3d w = x.head<3>();
    Vector3d v = x.tail<3>();
    Matrix3d w_skew;
    Matrix3d v_skew;

    skew(w,w_skew);
    skew(v,v_skew);

    y.block<3,3>(0,0) = w_skew;
    y.block<3,3>(3,0) = v_skew;
    y.block<3,3>(0,3) = Matrix3d::Zero();
    y.block<3,3>(3,3) = w_skew;
}

void se(Matrix<double, 6, 1> &x, Matrix<double, 4, 4> &y) {
    Vector3d w = x.head<3>();
    Vector3d v = x.tail<3>();
    Matrix3d w_skew;
    skew(w,w_skew);
    y.block<3,3>(0,0) = w_skew;
    y.block<3,1>(0,3) = v;
    y(3,0) = 0;
    y(3,1) = 0;
    y(3,2) = 0;
    y(3,3) = 0;

}

// initial implementation of rod
class Rod {
    private:
        double D;
        double L;
        double E;
        double G;
        double rho;
        double mu;
        double ds;
        int N;
        Matrix<double, 6, 6> K; //not necessarily diagonal, but for now its alright
        DiagonalMatrix<double, 6> M; //always diagonal
        DiagonalMatrix<double, 6> V; //I suppose not always diagonal
        Matrix<double, 6, 1> xi_ref;
    public:
        Matrix<double, Dynamic, 6> xi;
        Matrix<double, Dynamic, 6> eta;
        Matrix<double, Dynamic, 12> g;
        Rod(double D, double L, double E, double rho, double mu, int N);
        double energy(void);
        void step(double dt);
        VectorXd condition(Rod *prev, double dt, VectorXd xi0);
        void integrate(Rod *prev, double dt, VectorXd xi0);

};

// function object, for some reason called a functor in c++
class LM_Functor : public DenseFunctor<double> {
    double dt;
    Rod *prev;
    Rod &next;
    public:
        LM_Functor(Rod *r, Rod &c, double t) : DenseFunctor<double>(6, 6), prev(r), next(c), dt(t) {};
        int operator()(const VectorXd &x, VectorXd &fvec) const {
            // x is the input for xi0
            // fvec is the tip condition values
            fvec = next.condition(prev, dt, x);
            return 0;
        }
};

Rod::Rod(double d, double l, double e, double r, double m, int n) {
    D = d;
    L = l;
    E = e;
    G = E/3;
    rho = r;
    mu = m;
    N = n;
    ds = L/(N-1);

    double A = M_PI/4*pow(D,2);
    double I = M_PI/64*pow(D,4);
    double J = 2*I;

    //simplest initialization
    // seems like there would be a better way to initialize these
    int i;
    MatrixXd xi_temp(N,6);
    MatrixXd g_temp(N,12);
    for (i=0; i<N; i++) {
        xi_temp(i,0) = 0;
        xi_temp(i,1) = M_PI/(4*10e-2);
        xi_temp(i,2) = 0;
        xi_temp(i,3) = 0;
        xi_temp(i,4) = 0;
        xi_temp(i,5) = 1;

        g_temp(i,0) = 1;
        g_temp(i,1) = 0;
        g_temp(i,2) = 0;
        g_temp(i,3) = 0;
        g_temp(i,4) = 1;
        g_temp(i,5) = 0;
        g_temp(i,6) = 0;
        g_temp(i,7) = 0;
        g_temp(i,8) = 1;
        g_temp(i,9) = 0;
        g_temp(i,10) = 0;
        g_temp(i,11) = ds*i;
    }
    xi = xi_temp;
    g = g_temp;
    eta = MatrixXd::Constant(N,6,0);

    K = Matrix<double, 6, 6>::Zero();

    K.diagonal() << E*I, E*I, G*J, G*A, G*A, E*A;
    M.diagonal() << I, I, J, A, A, A;
    M = rho*M;
    V.diagonal() << 3*I, 3*I, J, A, A, 3*A;
    V = mu*V;

    xi_ref << 0, 0, 0, 0, 0, 1;

}

double Rod::energy(void) {
    double H = 0;
    int i;
    for (i=0; i<N; i++) {
        H += 0.5 * eta.row(i) * M * eta.row(i).transpose();
        H += 0.5 * (xi.row(i).transpose() - xi_ref).transpose() * K * (xi.row(i).transpose() - xi_ref);
    }
    return ds*H;
}

void Rod::step(double dt) {
    // solve the condition
    Rod prev = *this;
    VectorXd guess = xi.row(0);
    LM_Functor functor(&prev, *this, dt);
    DenseIndex nfev;
    int info;
    info = LevenbergMarquardt<LM_Functor>::lmdif1(functor, guess, &nfev);

//    this->integrate(&prev, dt, guess);
}

VectorXd Rod::condition(Rod *prev, double dt, VectorXd xi0) {

    this->integrate(prev, dt, xi0);
    return xi.row(N-1).transpose() - xi_ref;

}

void Rod::integrate(Rod *prev, double dt, VectorXd xi0) {

    Matrix<double, 6, 1> xi_half;
    Matrix<double, 6, 1> eta_half;
    Matrix<double, 6, 1> xi_half_next;
    Matrix<double, 6, 1> eta_half_next;
    Matrix<double, 6, 1> xi_dot;
    Matrix<double, 6, 1> eta_dot;
    Matrix<double, 6, 1> xi_der;
    Matrix<double, 6, 1> eta_der;
    Matrix<double, 6, 1> B_bar;

    Matrix<double, 6, 6> A_bar;
    Matrix<double, 6, 6> xi_ad;
    Matrix<double, 6, 6> eta_ad;

    Matrix4d eta_se;
    Matrix<double, 1, 12> g_row;

    xi.row(0) = xi0.transpose();

    Matrix4d G = Matrix4d::Identity();
    int i;
    for (i=0; i<N-1; i++) {
        xi_half = (xi.row(i) + prev->xi.row(i)).transpose()/2;
        eta_half = (eta.row(i) + prev->eta.row(i)).transpose()/2;

        xi_dot = (xi.row(i) - prev->xi.row(i)).transpose()/dt;
        eta_dot = (eta.row(i) - prev->eta.row(i)).transpose()/dt;

        A_bar = Matrix<double, 6, 6>::Zero();
        B_bar = Matrix<double, 6, 1>::Zero();

        B_bar = V * xi_dot;

        adjoint(xi_half, xi_ad);
        adjoint(eta_half, eta_ad);

        xi_der = K.ldlt().solve(((M * eta_dot) - (eta_ad.transpose() * M * eta_half) + (xi_ad.transpose() * K * (xi_half - xi_ref)) + B_bar));
        eta_der = xi_dot - xi_ad * eta_half;

        xi_half_next = xi_half + ds * xi_der;
        eta_half_next = eta_half + ds * eta_der;

        xi.row(i+1) = (2 * xi_half_next.transpose() - prev->xi.row(i+1));
        eta.row(i+1) = (2 * eta_half_next.transpose() - prev->eta.row(i+1));

    }

//    for (i=0; i<N; i++) {
//        g_row = prev->g.row(i);
//        unflatten(g_row, G);
//        eta_half = (eta.row(i)+prev->eta.row(i)).transpose()/2*ds;
//        se(eta_half, eta_se);
//        G = G * eta_se.exp();
//        flatten(G, g_row);
//        g.row(i) = g_row;
//    }

}





int main() {
    double dt = 0.05;
    Rod r1(1e-2,10e-2,1e6,1e3,0,40);
    cout << r1.energy() << endl;
    int i;
    for (i=0; i<1000; i++) {
        r1.step(dt);
        cout << r1.energy() << endl;
    }

//    Matrix<double, 6,1> xi;
//    xi << 1, 2, 3, 4, 5, 6;
//    Matrix4d xi_se;
//    Matrix<double, 6, 6> xi_ad;
//
//    cout << xi << endl;
//    se(xi,xi_se);
//    cout << xi_se << endl;
//    adjoint(xi,xi_ad);
//    cout << xi_ad << endl;


    return 0;
}