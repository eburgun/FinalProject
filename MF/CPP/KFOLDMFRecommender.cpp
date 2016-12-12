


#include "KFOLDMFRecommender.h"

KFOLDMFRecommender::KFOLDMFRecommender(int kValue, double lambda, double epsilon, int maxIter)
{
    kVal = kValue;
    lambdaVal = lambda;
    epsVal = epsilon;
    iterations = maxIter;
    srand(time(NULL));
}


void KFOLDMFRecommender::createPandQ(CSR * trainingSet)
{
    pMatrix = new double*[trainingSet->rows];
    qMatrix = new double*[trainingSet->columns];
    for(int i = 0; i < trainingSet->rows; i++){
        pMatrix[i] = new double[kVal];
        for(int j = 0; j < kVal; j++){
            pMatrix[i][j] = (double)rand() / (double)RAND_MAX;
        }

    }
    for(int i = 0; i < trainingSet->columns; i++){
        qMatrix[i] = new double[kVal];
        for(int j = 0; j< kVal; j++){
            qMatrix[i][j] = (double)rand() / (double)RAND_MAX;
        }
    }
}

void KFOLDMFRecommender::cleanUpPandQ(CSR * trainingSet)
{
    for(int i = 0; i<trainingSet->rows; i++){
        delete [] pMatrix[i];
        pMatrix[i] = nullptr;
    }
    for(int i = 0; i<trainingSet->columns; i++){
        
        delete [] qMatrix[i];
        qMatrix[i] = nullptr;
    }
    delete [] pMatrix;
    pMatrix = nullptr;
    delete [] qMatrix;
    qMatrix = nullptr;
}

void KFOLDMFRecommender::changeKValue(int newK, CSR * trainingSet)
{
    kVal = newK;
    cleanUpPandQ(trainingSet);
    createPandQ(trainingSet);
}

void KFOLDMFRecommender::changeLambda(double newLambda)
{
    lambdaVal = newLambda;
}

double KFOLDMFRecommender::fFunction(CSR * trainingSet)
{
    double pNorm = fNorm(pMatrix, trainingSet->rows);
    double qNorm = fNorm(qMatrix, trainingSet->columns);
    
    double lambdaQuantity = (pNorm + qNorm) * lambdaVal;
    double fSum = 0.0;
    for(int i = 0; i < trainingSet->rows;i++)
    {
        for(int j = 0; j <  trainingSet->columns;j++)
        {
            fSum += pow((trainingSet->getElement(i,j) - funcDotProduct(pMatrix[i],qMatrix[j])),2);
        }
    }
    return fSum + lambdaQuantity;
}

double KFOLDMFRecommender::funcDotProduct(double * a, double * b)
{
    double product = 0.0;
    for(int i = 0; i < kVal; i++)
    {
        product += a[i]*b[i];
    }
    return product;
}

double KFOLDMFRecommender::fNorm(double ** matrix,int dimension)
{
    double norm = 0.0;
    for(int i = 0; i <  dimension; i++){
        for(int j = 0; j < kVal; j++){
            norm += pow(matrix[i][j],2);
        }
    }
    return norm;
}

void KFOLDMFRecommender::LS_GD(CSR * dataSet, double ** fixedMatrix, double ** solvingMatrix,double learningRate,std::string matrixId)
{
    double lambdaValue = 1 - (lambdaVal * learningRate * 2);
    for(int i = 0; i < dataSet->rows;i++){
        double newItem[kVal];
        double sumMatrix[kVal];
        for(int j = dataSet->rowPtr[i]; j < dataSet->rowPtr[i+1]; j++){
            double dotProduct = 0.0;
            if(matrixId == "p"){
                dotProduct = funcDotProduct(solvingMatrix[i],fixedMatrix[dataSet->columnIndex[j]]);
            }
            else if(matrixId == "q"){
                dotProduct = funcDotProduct(fixedMatrix[dataSet->columnIndex[j]],solvingMatrix[i]);
            }
            double sumMult = (dataSet->ratingVals[j] - dotProduct);
            for(int k = 0; k < kVal; k++){
                sumMatrix[k] = fixedMatrix[dataSet->columnIndex[j]][k] * sumMult;
                
            }
        }
        double * newP = new double[kVal];
        for(int j = 0; j < kVal; j++){
            newItem[j] = solvingMatrix[i][j] * lambdaValue;
            sumMatrix[j] = sumMatrix[j] * (learningRate * 2);
            newP[j] = sumMatrix[j] + newItem[j];
            
        }
        double * temp = solvingMatrix[i];
        solvingMatrix[i] = newP;
        delete [] temp;
        temp = nullptr;
    }
}

void KFOLDMFRecommender::trainSystem(CSR * trainingSet, CSR * transposeSet)
{
    int i = 0;
    double lastIter = 0.0;

    while(i <  iterations){
        
        LS_GD(trainingSet, qMatrix, pMatrix, 0.025, "p");
        
        LS_GD(transposeSet, pMatrix, qMatrix, 0.025, "q");
        
        double curIter = fFunction(trainingSet);
        
        if(i > 0 && sqrt(pow((curIter - lastIter),2)/lastIter) < epsVal){
            break;
        } else {
          lastIter = curIter;
        }
        i++;
        
    }
    

}

double KFOLDMFRecommender::mSE(CSR * testingSet)
{
    double mse = 0.0;
    for(int i = 0; i < testingSet->rows;i++){
        for(int j = testingSet->rowPtr[i]; j < testingSet->rowPtr[i+1]; j++){
            double prediction = funcDotProduct(pMatrix[i],qMatrix[testingSet->columnIndex[j]]);
            mse += pow(testingSet->ratingVals[j] - prediction,2);
        }
    }

    mse /= testingSet->nonZeroValues;
    return mse;
}

double KFOLDMFRecommender::rMSE(double mse)
{
    double rmse = sqrt(mse);
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

void KFOLDMFRecommender::kFoldsTest(std::string trainStart, std::string testStart, std::string coldStart)
{
    std::ofstream outfile ("kFoldsResults.txt");
    for(int i = 1; i < 6; i++)
    {
        CSR * trainingSet = new CSR(trainStart + std::to_string(i) +".txt");
        CSR * transposeSet = new CSR(trainStart + std::to_string(i) + ".txt");
        transposeSet->transpose();
        CSR * testingSet = new CSR(testStart + std::to_string(i) + ".txt");
        CSR * coldSet = new CSR(coldStart + std::to_string(i) + ".txt");
        createPandQ(trainingSet);
        
        clock_t trainStart = clock();
        trainSystem(trainingSet, transposeSet);
        
        clock_t trainFinish = clock();
        clock_t testStart = clock();
        double mse = mSE(testingSet);
        double rmse = rMSE(mse);
        //double averageUser = createAverageUser();
        //coldStartTesting(coldSet, averageUser);
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
        outfile << (double)(trainFinish - trainStart)/CLOCKS_PER_SEC;
        outfile << " ";
        outfile << (double)(testFinish - testStart)/CLOCKS_PER_SEC;
        outfile << "\n";
        cleanUpPandQ(trainingSet);
        
        delete trainingSet;
        
        delete transposeSet;
        
        delete testingSet;
        
        delete coldSet;
        
    }
    outfile.close();
}

double KFOLDMFRecommender::createAverageUser(CSR * trainingSet)
{
    double averageUser[kVal];
    for(int i = 0; i < trainingSet->rows; i++)
    {
        for(int j = 0; j < kVal; j++)
        {
            averageUser[j] += pMatrix[i][j];
        }
    }
    for(int i = 0; i < kVal; i++)
    {
        averageUser[i] /= trainingSet->rows;
    }
    return averageUser;
}
/*
void KFOLDMFRecommender::coldStartTesting(CSR * coldSet, double * averageUser)
{
    double learningRate = 0.025;
    double ** newUserMatrix = new double*[coldSet->rows];
    double lambdaValue = 1 - (lambdaVal * learningRate * 2);
    for(int i = 0; i < coldSet->rows; i++){
        newUserMatrix[i] = new double[kVal];
        for(int j = 0; j < kVal; j++){
            newUserMatrix[i][j] = averageUser[j];
        }
        double newItem[kVal];
        double sumMatrix[kVal];
        for(int j = coldSet->rowPtr[i]; j < coldSet->rowPtr[i+1]; i++)
        {
            double ratingPrediction = funcDotProduct(newUserMatrix[i],qMatrix[coldSet->columnIndex[j]]);
            
            double sumMult = (coldSet->ratingVals[j] - dotProduct);
            for(int k = 0; k < kVal; k++){
                sumMatrix[k] = qMatrix[coldSet->columnIndex[j]][k] * sumMult;
            }
            //evaluate prediction
            //trainNewUserMatrix
        }
        double * newP = new double[kVal];
        for(int j = 0; j < kVal; j++){
            newItem[j] = newUserMatrix[i][j] * lambdaValue;
            sumMatrix[j] = sumMatrix[j] * (learningRate * 2);
            newP[j] = sumMatrix[j] + newItem[j];
            
        }
        double * temp = newUserMatrix[i];
        coldSet[i] = newP;
        delete temp;
        
    }
    //determine how quickly you get to ideal rating prediction

}



    Still need to create formal ratings
*/
