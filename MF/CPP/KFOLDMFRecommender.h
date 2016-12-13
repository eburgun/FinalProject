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
        void coldStartTesting(CSR * coldSet, double * averageUser);
        double * createAverageUser(CSR * trainingSet);
        void LS_GD(CSR * dataSet, double ** fixedMatrix, double ** solvingMatrix,double learningRate, std::string matrixId);
};

#include "KFOLDMFRecommender.cpp"
#endif
