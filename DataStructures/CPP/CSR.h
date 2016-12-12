//
//  CSR.h
//
//
//  Created by Evan Burgun on 10/3/16.
//
//

#ifndef CSR_h
#define CSR_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>


class CSR
{
    public:
        CSR (void);
        CSR (std::string fileName);
        ~CSR (void);
        void transpose(void);
        int getElement(int row, int col);
        int rows;
        int columns;
        int nonZeroValues;
        int * columnIndex;
        int * rowPtr;
        int * ratingVals;

    private:
        void processLine(std::string line,int increment);


};
#include "CSR.cpp"
#endif /* CSR_h */
