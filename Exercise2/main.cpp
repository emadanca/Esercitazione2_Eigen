#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

VectorXd palu_solver(const MatrixXd& A, const VectorXd& b) {
    PartialPivLU<MatrixXd> lu(A);
    VectorXd x = lu.solve(b);
    return x;
}

VectorXd qr_solver(const MatrixXd& A, const VectorXd& b) {
    HouseholderQR<MatrixXd> qr(A);
    VectorXd x = qr.solve(b);
    return x;
}

double relative_error(const VectorXd& x, const VectorXd& x_exact) {
    return (x - x_exact).norm() / x_exact.norm();
}

int main() {
    vector<pair<MatrixXd, VectorXd>> systems;

    // System 1
    MatrixXd A1(2, 2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    VectorXd b1(2);
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    systems.push_back(make_pair(A1, b1));

    // System 2
    MatrixXd A2(2, 2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    VectorXd b2(2);
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    systems.push_back(make_pair(A2, b2));

    // System 3
    MatrixXd A3(2, 2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    VectorXd b3(2);
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    systems.push_back(make_pair(A3, b3));

    VectorXd x_exact(2);
    x_exact << -1.0e+0, -1.0e+0;

    for (size_t i = 0; i < systems.size(); ++i) {
        auto& system = systems[i];
        VectorXd x_palu = palu_solver(system.first, system.second);
        VectorXd x_qr = qr_solver(system.first, system.second);
        double error_palu = relative_error(x_palu, x_exact);
        double error_qr = relative_error(x_qr, x_exact);
        cout << "System " << i+1 << ":\n";
        cout << "PALU Solution:\n" << x_palu << endl;
        cout << "QR Solution:\n" << x_qr << endl;
        cout << "PALU Relative Error: " << error_palu << endl;
        cout << "QR Relative Error: " << error_qr << endl;
        cout << endl;
    }

    return 0;
}
