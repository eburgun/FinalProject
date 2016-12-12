//
//  CSR.cpp
//
//
//  Created by Evan Burgun on 10/3/16.
//
//

#include "CSR.h"


CSR::CSR(void)
{
    rows = 0;
    columns = 0;
    nonZeroValues = 0;
    columnIndex = new int[0];
    rowPtr = new int[0];
    ratingVals = new int[0];   
}

CSR::CSR(std::string fileName)
{
    std::ifstream file(fileName.c_str());
    std::string line;
    int increment = 0;
    while (std::getline(file,line)) {
        if(increment == 0) {
            std::istringstream input(line);
            while(!input.eof()) {
                input >> rows;
                input >> columns;
                input >> nonZeroValues;
                
            }
            rowPtr = new int[rows+1];
            columnIndex = new int[nonZeroValues];
            ratingVals = new int[nonZeroValues];
            rowPtr[0] = 0;
        } else {
            
            processLine(line, increment);
        }
        increment++;
    }

}



CSR::~CSR(void)
{
    delete [] rowPtr;
    delete [] columnIndex;
    delete [] ratingVals;

}

void CSR::transpose(void)
{
    
    int nRow = columns;
    int nCol = rows;
    int * nRowPtr = new int[columns + 1];
    int rowCount[columns];
    int * nColIdx = new int[nonZeroValues];
    int * nValues = new int[nonZeroValues];
    
    for(int i = 0; i < columns + 1; i++) {
        nRowPtr[i] = 0;
    }
    for(int i = 0; i < columns; i++) {
        rowCount[i] = 0;
    }
    
    for(int i = 0; i < nonZeroValues; i++) {
        nRowPtr[columnIndex[i] + 1] += 1;
    }
    for(int i = 0; i < columns; i++) {

        nRowPtr[i+1] += nRowPtr[i];

    }
    
    for(int i = 0; i < rows; i++) {
        for(int j = rowPtr[i];j < rowPtr[i+1];j++) {

            int i2 = columnIndex[j];

            nColIdx[nRowPtr[i2] + rowCount[i2]] = i;

            nValues[nRowPtr[i2] + rowCount[i2]] = ratingVals[j];

            rowCount[i2] += 1;

        }
    }
    
    
    int * temp1 = rowPtr;
    int * temp2 = columnIndex;
    int * temp3 = ratingVals;
    rowPtr = nRowPtr;
    columnIndex = nColIdx;
    ratingVals = nValues;
    rows = nRow;
    columns = nCol;
    delete[] temp1;
    delete[] temp2;
    delete[] temp3;

}


void CSR::processLine(std::string line,int increment)
{
    std::istringstream input(line);
    int i = 0;
    int counter = 0;
    std::string newLine;
    while(std::getline(input,newLine,' ')) {
        if(i%2 == 0) {
            columnIndex[rowPtr[increment-1]+i/2] = atoi(newLine.c_str())-1;
        } else {
            ratingVals[rowPtr[increment-1]+(i-1)/2] = atoi(newLine.c_str());
        }
        i++;
    }
    rowPtr[increment] = i/2 + rowPtr[increment-1];
}

int CSR::getElement(int row, int col)
{
    int element = 0;
    if(row < rows && col < columns) {
        for(int i = rowPtr[row]; i < rowPtr[row+1];i++) {
            if(columnIndex[i] == col) {
                element = ratingVals[i];
            }
        }
    } else {
        std::cout << "Value not in matrix" << std::endl;
    }
    
    return element;
}

