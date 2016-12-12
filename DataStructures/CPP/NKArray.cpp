#include "NKArray.h"

NKArray::NKArray(int items,int kValue)
{
    rows = items;
    columns = 2*kValue;
    similarities = new float*[rows];
    index = new int*[rows];
    for (int i = 0; i < rows; i++)
    {
        similarities[i] = new float[columns];
        index[i] = new int[columns];
        for(int j = 0; j < columns; j++)
        {
            similarities[i][j] = 0.0;
            index[i][j] = 0;
        }
    }
}

NKArray::~NKArray(void)
{
    for(int i = 0; i < rows; i++)
    {
        
        delete [] similarities[i];
        delete [] index[i];
        similarities[i] = nullptr;
        index[i] = nullptr;
        
    }
    delete [] similarities;
    delete [] index;
    similarities = nullptr;
    index = nullptr;
}

void NKArray::insertElement(int x, float simil, int ind)
{

    similarities[x][columns - 1] = simil;
    index[x][columns - 1] = ind;
    
    int i = (columns) - 1;
    while(i > 0 && similarities[x][i] > similarities[x][i-1])
    {
        float tempS = similarities[x][i-1];
        int tempI = index[x][i-1];
        similarities[x][i-1] = similarities[x][i];
        index[x][i-1] = index[x][i];
        similarities[x][i] = tempS;
        index[x][i] = tempI;
        i -= 1;
    }
}
float NKArray::getSimil(int x, int y)
{
    return similarities[x][y];
}
int NKArray::getIndex(int x, int y)
{
    return index[x][y];
}