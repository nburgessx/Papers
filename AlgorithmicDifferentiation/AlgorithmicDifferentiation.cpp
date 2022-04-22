// Program to demonstrate tangent and adjoint differentiation for a simple function f(x1,x2) = 2.x1^2 + 3.x2
#include <vector>
#include <iostream>
using namespace std;

// Method to Display the Target Function and Expected Results
void showFunction(const vector<double> & x)
{
    cout << "Function" << endl;
    cout << "f(x1, x2) = 2.x1^2 + 3.x1" << endl;
    cout << "df/dx1 = 4.x1" << endl;
    cout << "df/dx2 = 3" << endl;
    cout << endl;

    if (x.size()!=2) return;
    double x1 = x[0];
    double x2 = x[1];
    
    cout << "Expected Results" << endl;
    cout << "x1 = " << x1 << endl;
    cout << "x2 = " << x2 << endl;
    cout << "f(x1,x2) = " << 2*x1*x1 + 3*x2 << endl;
    cout << "df/dx1 = " << 4*x1 << endl;
    cout << "df/dx2 = " << 3 << endl;
    cout << endl;
}

// Forward 'Tangent' Method
// Risk variables are denoted 'dot' with '_d' suffix
// [IN] vector of input variables x (x1, x2)
// [IN] vector of risk activation variables x_d (x1_d, x2_d)
void tangent(const vector<double> & x, const vector<double> & x_d)
{
    // Target Function: f(x1,x2) = 2.x1^2 + 3.x2
    
    // Inputs: x1 and x2
    if (x.size()!=2) return;
    double x1 = x[0];
    double x2 = x[1];
    
    // Risk Activation Inputs: x1_d and x2_d
    if (x_d.size()!=2) return;       
    double x1_d = x_d[0];           // Init Tangent Mode
    double x2_d = x_d[1];           // Init Tangent Mode
    
    // Forward Calculation
    double a = x1 * x1;             // 1.   a = x1^2
    double a_d = 2 * x1 * x1_d;
    
    double b = 2 * a;               // 2.   b = 2.x1^2
    double b_d = 2 * a_d;     
    
    double c = x2;                  // 3.   c = x2
    double c_d = x2_d;
    
    double d = 3 * c;               // 4.   d = 3.x2
    double d_d = 3*c_d;
    
    double f = b + d;               // 5.   f = 2.x1^2 + 3.x2
    double f_d = b_d + d_d;
    
    // Tangent Results
    
    // Tangent Mode calculates one input sensitivity at a time (similar to numerical bumping)
    if(x1_d==1.0 && x2_d==0.0)
    {
        cout << "Tangent Results for x1" << endl;
        cout << "x1 = " << x1 << endl;
        cout << "x2 = " << x2 << endl;
        cout << "f(x1,x2) = " << f << endl;
        cout << "df/dx1 = " << f_d << endl;
        cout << endl;
    }
    
    // Tangent Mode calculates one input sensitivity at a time (similar to numerical bumping)        
    if(x1_d==0.0 && x2_d==1.0)
    {
        cout << "Tangent Results for x2" << endl;
        cout << "x1 = " << x1 << endl;
        cout << "x2 = " << x2 << endl;
        cout << "f(x1,x2) = " << f << endl;
        cout << "df/dx2 = " << f_d << endl;
        cout << endl;
    }
    
}

// Backward 'Adjoint' Method
// Risk variables are denoted 'bar' with '_b' suffix
// [IN] vector of input variables x (x1, x2)
// [IN] vector of risk activation variables f_b (f1_b, f2_b)
void adjoint(const vector<double> & x, const vector<double> & f_b)
{
    // Target Function: f(x1,x2) = 2.x1^2 + 3.x2
    
    // Inputs: x1 and x2
    if (x.size()!=2) return;
    double x1 = x[0];
    double x2 = x[1];
    
    // Risk Activation Inputs: f1_b and f2_b
    if (f_b.size()!=2) return;       
    double f1_b = f_b[0];           // Init Adjoint Mode
    double f2_b = f_b[1];           // Init Adjoint Mode
    
    // Forward Sweep
    double a = x1 * x1;             // 1.   a = x1^2
    double b = 2 * a;               // 2.   b = 2.x1^2
    double c = x2;                  // 3.   c = x2
    double d = 3 * c;               // 4.   d = 3.x2
    double f = b + d;               // 5.   f = 2.x1^2 + 3.x2
    
    // Back Propogation
    double b_b = f1_b;              // 5.
    double d_b = f2_b;              // 5.
    double c_b = 3 * d_b;           // 4.
    double x2_b = c_b;              // 3.
    double a_b = 2 * b_b;           // 2.
    double x1_b = 2 * x1 * a_b;     // 1.
    
    // Adjoint Results
    cout << "Adjoint Results" << endl;
    cout << "x1 = " << x1 << endl;
    cout << "x2 = " << x2 << endl;
    cout << "f(x1,x2) = " << f << endl;
    cout << "df/dx1 = " << x1_b << endl;
    cout << "df/dx2 = " << x2_b << endl;
    cout << endl;
}

int main()
{
    vector<double> x = {2, 2};
    showFunction(x);
    
    // Note: Tangent Mode calculates one input sensitivity at a time (similar to numerical bumping)  
    
    // Tangent mode for x1_d
    vector<double> x1_d = {1,0};
    tangent(x, x1_d);
    
    // Tangent mode for x2_d
    vector<double> x2_d = {0,1};
    tangent(x, x2_d);
    
    vector<double> f_b = {1,1};
    adjoint(x, f_b);
    
    return 0;
}
