/*
 * MFRecommender.h
 *
 * Evan Burgun
 * Programming Assignment 2
 *
 */

#ifndef MFRecommender_h
#define MFRecommender_h

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include "CSR.h"

class MFRecommender
{
    public:
        MFRecommender(std::string trainFile, std::string testFile, int kValue, float lambda, float epsilon, int maxIter);
        ~MFRecommender(void);
        void changeKValue(int newK);
        void changeLambda(float newLambda);
        void trainSystem(void);
        float testMSE(void);
        float testSet(float mse);
        void testingMethod(void);

    private:
        int kVal;
        float lambdaVal;
        float epsVal;
        int iterations;
        CSR * trainingData;
        CSR * testingData;
        float ** pMatrix;
        float ** qMatrix;
        void createPandQWithRandom(void);
        void cleanUpPandQ(void);
        float fFunction(void);
        float funcDotProduct(float * a, float * b);
        float fNorm(float ** matrix,int dimension);

        void LS_GD(CSR * dataSet, float ** fixedMatrix, float ** solvingMatrix,float learningRate, std::string matrixId);
};


#include "MFRecommender.cpp"
#endif
