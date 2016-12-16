#ifndef KFOLDMFRecommender_h
#define KFOLDMFRecommender_h

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <numeric>
#include <time.h>
#include <iostream>
#include <fstream>
#include "../../DataStructures/CPP/CSR.h"

class KFOLDMFRecommender
{
    public:
        KFOLDMFRecommender(int kValue, double lambda, double epsilon, int maxIter);
        
        void changeKValue(int newK, CSR * trainingSet);
        void changeLambda(double newLambda);
        void kFoldsTest(std::string trainStart, std::string testStart, std::string coldStart);
        
    private:
        int kVal;
        double lambdaVal;
        double epsVal;
        int iterations;
        double ** pMatrix;
        double ** qMatrix;
        double funcDotProduct(double * a, double * b);
        double fNorm(double ** matrix,int dimension);
        void createPandQ(CSR * trainingSet);
        void cleanUpPandQ(CSR * trainingSet);
        double fFunction(CSR * trainingSet);
        void trainSystem(CSR * trainingSet, CSR * transposeSet);
        double mSE(CSR * testingSet);
        double rMSE(double mse);
        void coldStartTesting(CSR * coldSet, double * averageUser,CSR * trainingSet, std::ofstream& outfile);
        double * createAverageUser(CSR * trainingSet);
        double ** predictUserRecs(double * user, int nItems,int ** trained, int sizeOfTrained, CSR * trainingSet);
        void quickSort(double ** sArray, int arraySize,int index);
        void part(double ** sArray, int left, int right,int index);
        double pivot(double ** sArray, int left, int right,int index);
        void swapper(double ** sArray, int a, int b);
        void insertionSort(double ** sArray, int left, int right,int index);
        void trainSingleUser(double * userMatrix, int usersItem, int usersRating);
        double cosSimil(double * a, double * b);
        void LS_GD(CSR * dataSet, double ** fixedMatrix, double ** solvingMatrix,double learningRate, std::string matrixId);
        void testingMethod(CSR * trainingSet, CSR * transposeSet, CSR * testingSet, CSR * coldSet, std::ofstream& outfile);
};

#include "KFOLDMFRecommender.cpp"
#endif
