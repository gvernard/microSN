#include <iostream>

int main(int argc, char* argv[])
{
    double m1, m2, v;

    while(true)
    {
        std::cin >> m1 >> m2 >> v;        
        std::cout << "Received m1, m2, v = " << m1 << ", " << m2 << ", " << v << ", outputting: " << m1 + m2 + v << std::endl;
        std::cout.flush();
    }
}