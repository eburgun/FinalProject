


#include "KFOLDMFRecommender.h"

KFOLDMFRecommender::KFOLDMFRecommender(int kValue, double lambda, double epsilon, int maxIter):
    kVal(kValue), lambdaVal(lambda), epsVal(epsilon),iterations(maxIter)
{
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
        
    }
    for(int i = 0; i<trainingSet->columns; i++){
        
        delete [] qMatrix[i];
       
    }
    delete [] pMatrix;
    
    delete [] qMatrix;
   
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
    std::cout << "k = " + std::to_string(kVal) + " lambda = " + std::to_string(lambdaVal) + " maxIters = " + std::to_string(iterations) + " epsilon = " + std::to_string(epsVal) + " mse = " + std::to_string(mse) + " rmse = " + std::to_string(rmse) << std::endl;
    return rmse;
}

void KFOLDMFRecommender::kFoldsTest(std::string trainStart, std::string testStart, std::string coldStart)
{
    std::ofstream outfile ("kFoldsResults.txt");
    
        
        CSR * trainingSet = new CSR(trainStart + std::to_string(1) +".txt");
        CSR * transposeSet = new CSR(trainStart + std::to_string(1) + ".txt");
        transposeSet->transpose();
        CSR * testingSet = new CSR(testStart + std::to_string(1) + ".txt");
        CSR * coldSet = new CSR(coldStart + std::to_string(1) + ".txt");
        createPandQ(trainingSet);
        clock_t trainStartTime = clock();
        trainSystem(trainingSet, transposeSet);
        clock_t trainFinish = clock();
        std::cout << std::to_string((double)(trainFinish - trainStartTime)/CLOCKS_PER_SEC) + " ";
        clock_t testStartTime = clock();
        double mse = mSE(testingSet);
        double rmse = rMSE(mse);
        
        double * averageUser = createAverageUser(trainingSet);
        /*
        int * trained = new int[trainingSet->columns];
        for(int i = 0; i < 20; i++)
        {
            trained[i] = 0;
        }
        double ** userRecs = predictUserRecs(averageUser, trainingSet->columns, trained);
        for(int i = 0; i < 20; i ++)
        {
            std::cout << std::to_string(userRecs[i][0]) + " " + std::to_string(userRecs[i][1]) << std::endl;
        }
        */
        coldStartTesting(coldSet, averageUser);
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
        outfile << (double)(trainFinish - trainStartTime)/CLOCKS_PER_SEC;
        outfile << " ";
        outfile << (double)(testFinish - testStartTime)/CLOCKS_PER_SEC;
        outfile << "\n";
        cleanUpPandQ(trainingSet);
        
        
        delete trainingSet;
        
        delete transposeSet;
        
        delete testingSet;
        
        delete coldSet;
        
        delete averageUser;
        /*
        for(int i = 0; i < 20; i ++)
        {
            delete [] userRecs[i];
        }
        delete [] userRecs;
        */
    outfile.close();
}

double * KFOLDMFRecommender::createAverageUser(CSR * trainingSet)
{
    double * averageUser = new double[kVal];
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

void KFOLDMFRecommender::coldStartTesting(CSR * coldSet, double * averageUser)
{
    double ** newUserMatrix = new double*[coldSet->rows];
    double totalHR = 0.0;
    for(int i = 0; i < coldSet->rows; i++)
    {
        double bestPotHit = 0.0;
        int maxHits = 0;
        newUserMatrix[i] = new double[kVal];
        for(int j = 0; j < kVal; j++)
        {
            newUserMatrix[i][j] = averageUser[j];
        }
        int upcomingItemsCount = coldSet->rowPtr[i+1] - coldSet->rowPtr[i];
        int * untrainedItem = new int[upcomingItemsCount];
        int * untrainedRatings = new int[upcomingItemsCount];
        int trainedItems = 0;
        for(int j = coldSet->rowPtr[i]; j < coldSet->rowPtr[i+1]; j++){
            untrainedItem[j-coldSet->rowPtr[i]] = coldSet->columnIndex[j];
            untrainedRatings[j-coldSet->rowPtr[i]] = coldSet->ratingVals[j];
        }
        while(trainedItems < upcomingItemsCount)
        {
            double ** userRecs = predictUserRecs(newUserMatrix[i], coldSet->columns, untrainedItem);
            int hits = 0; 
            for(int j = 0; j < upcomingItemsCount; j ++)
            {
                for(int k = 0; k < 20; k++)
                {
                    if(untrainedItem[j] == userRecs[k][1])
                    {
                        hits++;
                    }
                }
            }
            double HR = 0.0;
            if(upcomingItemsCount - trainedItems < 20)
            {
                HR = hits/upcomingItemsCount-trainedItems;
            }else
            {
                HR = hits/20.0;
            }
            if(HR >= bestPotHit)
            {
                bestPotHit = HR;
            }
            /*
             * Still need error reporting
             **/
            int randomValue = rand() % upcomingItemsCount;
            while(untrainedItem[randomValue] == -1)
            {
                randomValue = rand() % upcomingItemsCount;
            }
            int usersNext = untrainedItem[randomValue];
            untrainedItem[randomValue] = -1;
            
            trainedItems++;
            for(int j = 0; j < 20; j ++)
            {
                delete[] userRecs[j];
            }
            delete [] userRecs;
        }
        
        delete [] untrainedItem;
        delete [] untrainedRatings;
        totalHR += bestPotHit;
    }
    
    for(int i = 0; i < coldSet->rows; i++)
    {
        delete [] newUserMatrix[i];
    }
    delete [] newUserMatrix;
    std::cout << totalHR/coldSet->rows << std::endl;
    
/*
 *Method for testing cold start:
 *
 *For every user in ColdStartSet:
 *  while(user has [untrained Items]):
 *      Create set of recommendations for the user
 *      get predicted ratings for those recommendations
 *      create [hits] by determining which recommendations exist in [untrained items]
 *
 *      if(!hits.empty()):
 *          choose random item from [untrained]. (Consider weighting based on hits)
 *          Calculate HR, ARHR, Predicted Rating Error (accumulate with total metrics)
 *
 *      else: 
 *          choose random item from [untrained].
 *          //accumulate total, HR, ARHR
        train dataSet for new user and random item
        Save current change in hr, arhr, and error
        
    calculate mse, rmse
    calculate total hr, arhr for user
    
    //Also consider
    trained Items
    error per new item addition
    total hr, arhr, mse, rmse per user-addressed
    total hr, arhr, mse, rmse per userSet
    
    

 */    
}

double ** KFOLDMFRecommender::predictUserRecs(double * user, int nItems, int * trained)
{
    int n = 20;
    
    double ** predictedRatings= new double*[nItems];
    
    for(int i = 0; i < nItems; i++)
    {
        predictedRatings[i] = new double[2];
        if(trained[i] != -1)
        {
            
            predictedRatings[i][0] = funcDotProduct(user, qMatrix[i]);
            predictedRatings[i][1] = i;
        } else
        {
            predictedRatings[i][0] = 0.0;
            predictedRatings[i][-1] = -1.0;
        }

    }
    quickSort(predictedRatings, nItems);
    
    double ** nVals = new double * [n];
    
    int i = 0;
    int inNVals = 0;
    while(i < nItems && inNVals < n)
    {
        if(predictedRatings[i][1] != -1.0)
        {
            nVals[i] = new double[2];
            
            nVals[i][0] = predictedRatings[i][0];
            nVals[i][1] = predictedRatings[i][1];
            inNVals++;
        }
        i++;
    }
    if(inNVals < n)
    {
        for(int i = inNVals; i < n; i++)
        {
            nVals[i] = new double[2];
            nVals[i][0] = 0.0;
            nVals[i][1] = -1;
        }
    }
    delete [] predictedRatings;
    
    return nVals;
}

void KFOLDMFRecommender::quickSort(double ** sArray, int arraySize)
{
    part(sArray, 0, arraySize - 1);
}

void KFOLDMFRecommender::part(double ** sArray, int left, int right)
{
    if((right - left) > 5)
    {
        double pivotValue = pivot(sArray, left, right);

        int i = left + 1;
        int j = right;

        while (i < j)
        {
            while(pivotValue > sArray[--j][0]){}
            while (sArray[++i][0] > pivotValue){}
            if(i < j)
            {
                swapper(sArray,i,j);
            }
        }
        swapper(sArray,left + 1, j);

        part(sArray,left,j-1);
        part(sArray,j+1,right);
    }
    else
    {
        insertionSort(sArray,left,right);
    }

}
double KFOLDMFRecommender::pivot(double ** sArray,int left,int right)
{
    int center = (left + right) / 2;

    if (sArray[center][0] < sArray[right][0])
    {
        swapper(sArray,right,center);
    }
    if (sArray[left][0] < sArray[right][0])
    {
        swapper(sArray,left,right);
    }
    if (sArray[left][0] < sArray[center][0])
    {
        swapper(sArray,left,center);
    }

    swapper(sArray,center,(left + 1));
    return sArray[left + 1][0];
}

void KFOLDMFRecommender::swapper(double ** sArray, int a, int b)
{
    double * temp = sArray[a];
    sArray[a] = sArray[b];
    sArray[b] = temp;
}

void KFOLDMFRecommender::insertionSort(double ** sArray, int left, int right)
{
    int i;
    int j;
    for(i = left; i <= right; i ++)
    {
        double * tmp = sArray[i];
        j = i + left;
        for( j = i; j > left && tmp[0] > sArray[j-1][0]; j--)
        {
            sArray[j] = sArray[j-1];
        }
        sArray[j] = tmp;
    }
}