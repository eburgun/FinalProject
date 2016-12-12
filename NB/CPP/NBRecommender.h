#ifndef NBRecommender_h
#define NBRecommender_h

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include "../DataStructures/CSR.h"
#include "../DataStructures/NKArray.h"

class NBRecommender
{
    public:
        NBRecommender(std::string trainFile, std::string testFile, std::string outFileStr, int kVal, int nVal);
        ~NBRecommender(void);
        float*** recommendations(void);
        //void saveRecs(void);
        void changeNValue(int newN);
        void changeKValue(int newK);
        void changeOutfile(std::string newFile);
        void rankKVals(int * itemsArray, int itemsSize,float ** kList,int kListSize);
        void testingMethod(void);
        
    private:
        int nValue;
        int kValue;
        std::string outFile;
        CSR * trainingData;
        CSR * trainingTranspose;
        CSR * testingData;
        std::unordered_map <std::string, float> cosSimil;
        NKArray * nkArray;
        bool nkBuilt;
        bool kChanged;
        void buildNKArray(void);
        void rebuildNKArray(void);
        float * testRecsHR(float *** userRecs);
        float ** pullKTopVals(int userId);
        void insertionSort(float ** array, int left, int right);
        void swap(float ** array, int a, int b);
        int pivot(float ** array, int left, int right);
        void partition(float ** array, int left, int right);
        void quickSort(float ** array,int length);
        
        
    
};

#include "NBRecommender.cpp"
#endif
