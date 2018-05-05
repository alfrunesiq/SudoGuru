#include <iostream>

#include "sudoguru.hpp"

int main(int argc, char **argv)
{
    int exit_code = 0;
    try {
        exit_code = sudoguru(argc, argv);
    } catch (const std::exception e) {
        std::cerr << "Caught exception:\n"
                << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Caught unknown exception!\n" 
                << std::endl;
    }

    return exit_code;
}
