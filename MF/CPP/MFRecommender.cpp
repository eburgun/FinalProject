/*
 * MFRecommender.cpp
 *
 * Evan Burgun
 * Programming Assignment 2
 *
 */

#include "MFRecommender.h"

MFRecommender::MFRecommender(std::string trainFile, std::string testFile, int kValue, float lambda, float epsilon, int maxIter)
{
    kVal = kValue;
    lambdaVal = lambda;
    epsVal = epsilon;
    iterations = maxIter;
    trainingData = new CSR(trainFile);
    testingData = new CSR(testFile);
    srand(time(NULL));
    createPandQWithRandom();

}

MFRecommender::~MFRecommender(void)
{

    cleanUpPandQ();
    delete trainingData;
    delete testingData;


}

void MFRecommender::createPandQWithRandom(void)
{
    pMatrix = new float*[trainingData->rows];
    qMatrix = new float*[trainingData->columns];
    for(int i = 0; i < trainingData->rows; i++){
        pMatrix[i] = new float[kVal];
        for(int j = 0; j < kVal; j++){
            pMatrix[i][j] = (double)rand() / (double)RAND_MAX;
        }

    }
    for(int i = 0; i < trainingData->columns; i++){
        qMatrix[i] = new float[kVal];
        for(int j = 0; j< kVal; j++){
            qMatrix[i][j] = (double)rand() / (double)RAND_MAX;
        }
    }
}

void MFRecommender::cleanUpPandQ(void)
{
    for(int i = 0; i<trainingData->rows; i++){
        delete [] pMatrix[i];
    }
    for(int i = 0; i<trainingData->columns; i++){
        delete [] qMatrix[i];
    }
    delete [] pMatrix;
    delete [] qMatrix;
}

void MFRecommender::changeKValue(int newK)
{
    kVal = newK;
    cleanUpPandQ();
    createPandQWithRandom();
}

void MFRecommender::changeLambda(float newLambda)
{
    lambdaVal = newLambda;
}

float MFRecommender::fFunction(void)
{

    float pNorm = fNorm(pMatrix,trainingData->rows);
    float qNorm = fNorm(qMatrix,trainingData->columns);

    float lambdaQuantity = (pNorm + qNorm) * lambdaVal;
    float fSum = 0.0;
    for(int i = 0; i < trainingData->rows;i++)
    {
        for(int j = 0; j <  trainingData->columns;j++)
        {
            fSum += pow((trainingData->getElement(i,j) - funcDotProduct(pMatrix[i],qMatrix[j])),2);
        }
    }
    return fSum + lambdaQuantity;
}

float MFRecommender::funcDotProduct(float * a, float * b)
{
    float product = 0.0;
    for(int i = 0; i < kVal; i++)
    {
        product += a[i]*b[i];
    }
    return product;
}

float MFRecommender::fNorm(float ** matrix,int dimension)
{
    float norm = 0.0;
    for(int i = 0; i <  dimension; i++){
        for(int j = 0; j < kVal; j++){
            norm += pow(matrix[i][j],2);
        }
    }
    return norm;
}

void MFRecommender::LS_GD(CSR * dataSet, float ** fixedMatrix, float ** solvingMatrix,float learningRate,std::string matrixId)
{
    float lambdaValue = 1 - (lambdaVal * learningRate * 2);
    for(int i = 0; i < dataSet->rows;i++){
        float newItem[kVal];
        float sumMatrix[kVal];
        for(int j = dataSet->rowPtr[i]; j < dataSet->rowPtr[i+1]; j++){
            float dotProduct = 0.0;
            if(matrixId == "p"){
                dotProduct = funcDotProduct(solvingMatrix[i],fixedMatrix[dataSet->columnIndex[j]]);
            }
            else if(matrixId == "q"){
                dotProduct = funcDotProduct(fixedMatrix[dataSet->columnIndex[j]],solvingMatrix[i]);
            }
            float sumMult = (dataSet->ratingVals[j] - dotProduct);
            for(int k = 0; k < kVal; k++){
                sumMatrix[k] = fixedMatrix[dataSet->columnIndex[j]][k] * sumMult;
            }
        }
        float * newP = new float[kVal];
        for(int j = 0; j < kVal; j++){
            newItem[j] = solvingMatrix[i][j] * lambdaValue;
            sumMatrix[j] = sumMatrix[j] * (learningRate * 2);
            newP[j] = sumMatrix[j] + newItem[j];

        }
        float * temp = solvingMatrix[i];
        solvingMatrix[i] = newP;
        delete temp;
    }
}

void MFRecommender::trainSystem(void)
{
    int i = 0;
    float lastIter = 0.0;
    while(i <  iterations){
        LS_GD(trainingData, qMatrix, pMatrix, 0.025, "p");
        trainingData->transpose();
        LS_GD(trainingData, pMatrix, qMatrix, 0.025, "q");
        trainingData->transpose();
        float curIter = fFunction();
        if(i > 0 && sqrt(pow((curIter - lastIter),2)/lastIter) < epsVal){
            break;
        } else {
          lastIter = curIter;
        }
        i++;
    }
}

float MFRecommender::testMSE(void)
{
    float mse = 0.0;
    for(int i = 0; i < testingData->rows;i++){
        for(int j = testingData->rowPtr[i]; j < testingData->rowPtr[i+1]; j++){
            float prediction = funcDotProduct(pMatrix[i],qMatrix[testingData->columnIndex[j]]);
            mse += pow(testingData->ratingVals[j] - prediction,2);
        }
    }

    mse /= testingData->nonZeroValues;
    return mse;
}

float MFRecommender::testSet(float mse)
{
    float rmse = sqrt(mse);
    std::cout << "k = ";
    std::cout << kVal;
    std::cout << " lambda = ";
    std::cout << lambdaVal;
    std::cout << " maxIters = ";
    std::cout << iterations;
    std::cout << " epsilon = ";
    std::cout << epsVal;
    std::cout << " mse = ";
    std::cout << mse;
    std::cout << " rmse = ";
    std::cout << rmse << std::endl;
    return rmse;
}

void MFRecommender::testingMethod(void)
{
    int kVals [] = {10 , 50};
    float lambVals [] = {0.01,0.1,1,10};
    int iters [] = {50,100,200};
    float epsilonVals [] = {0.0001,0.001,0.01};
    std::ofstream outfile ("results.txt");
    for(int i = 0; i < 2; i++){
        kVal = kVals[i];
        for(int j = 0; j < 4; j++){
            lambdaVal = lambVals[j];
            for (int k = 0; k < 3; k++){
                iterations = iters[k];
                for (int l = 0; l < 3; l++){
                    epsVal = epsilonVals[l];
                    cleanUpPandQ();
                    createPandQWithRandom();
                    clock_t trainStart = clock();
                    trainSystem();
                    clock_t trainFinish = clock();
                    clock_t testStart = clock();
                    float mse = testMSE();
                    float rmse = testSet(mse);
                    clock_t testFinish = clock();
                    outfile << kVal;
                    outfile << " ";
                    outfile << lambdaVal;
                    outfile << " ";
                    outfile << iterations;
                    outfile << " ";
                    outfile << epsVal;
                    outfile << " ";
                    outfile << mse;
                    outfile << " ";
                    outfile << rmse;
                    outfile << " ";
                    outfile << (double)(trainFinish - trainStart) * 1000.0/CLOCKS_PER_SEC;
                    outfile << " ";
                    outfile << (double)(testFinish - testStart) * 1000.0/CLOCKS_PER_SEC;
                    outfile << "\n";

                }
            }
        }
    }

    outfile.close();
}
