#ifndef NKArray_h
#define NKArray_h

class NKArray
{
    public:
        NKArray(int items,int kValue);
        ~NKArray(void);
        void insertElement(int x, float simil, int ind);
        float getSimil(int x, int y);
        int getIndex(int x, int y);
    private:
        int rows;
        int columns;
        float ** similarities;
        int ** index;
};

#include "NKArray.cpp"
#endif