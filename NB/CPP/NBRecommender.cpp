

#include "NBRecommender.h"

NBRecommender::NBRecommender(std::string trainFile, std::string testFile, std::string outFileStr, int kVal, int nVal)
{
    kValue = kVal;
    nValue = nVal;
    outFile = outFileStr;
    trainingData = new CSR(trainFile);
    trainingTranspose = new CSR(trainFile);
    trainingTranspose->transpose();
    testingData = new CSR(testFile);
    nkBuilt = false;
    kChanged = false;
    buildNKArray();
    clock_t start = clock();
    pullKTopVals(0);
    //recommendations();
    std::cout << (clock() - start) / 100000 << std::endl;
    //testingMethod();

}

NBRecommender::~NBRecommender(void)
{
    cosSimil.clear();
    delete nkArray;
    nkArray = nullptr;
    delete trainingData;
    trainingData = nullptr;
    delete trainingTranspose;
    trainingTranspose = nullptr;
    delete testingData;
    testingData = nullptr;
    
}

void NBRecommender::changeNValue(int newN)
{
    nValue = newN;
}

void NBRecommender::changeKValue(int newK)
{

    delete nkArray;
    nkArray = nullptr;
    kValue = newK;
    kChanged = true;
}

void NBRecommender::changeOutfile(std::string newFile)
{
    outFile = newFile;
}

void NBRecommender::buildNKArray(void)
{
    
    int * lengthArray = new int[trainingTranspose->rows];
    float * cosArray = new float[trainingTranspose->rows];
    nkArray = new NKArray(trainingTranspose->rows,kValue);
    for(int i = 0; i < trainingTranspose->rows; i++)
    {
        for(int j = trainingTranspose->rowPtr[i]; j < trainingTranspose->rowPtr[i+1];j++)
        {
            lengthArray[i] += pow(trainingTranspose->ratingVals[j],2);
            
        }

        for(int j = trainingTranspose->rowPtr[i]; j < trainingTranspose->rowPtr[i+1]; j++)
        {
            for(int k = trainingData->rowPtr[trainingTranspose->columnIndex[j]]; k < trainingData->rowPtr[trainingTranspose->columnIndex[j] + 1];k++)
            {
                if(trainingData->columnIndex[k] != i){
                    cosArray[trainingData->columnIndex[k]] += trainingData->ratingVals[k] * trainingTranspose->ratingVals[j];
                }
            }
        }
        for(int j = 0; j < trainingTranspose->rows; j++)
        {
            if (lengthArray[i] != 0 && lengthArray[j] != 0)
            {
                cosArray[j] /= sqrt(lengthArray[i]) * sqrt(lengthArray[j]);
                if (i < j)
                {
                    if (cosSimil.find(std::to_string(i) + " " + std::to_string(j)) == cosSimil.end())
                    {
                        cosSimil[std::to_string(i) + " " + std::to_string(j)] = cosArray[j];
                    }
                }
                else if (j < i)
                {
                    if(cosSimil.find(std::to_string(j) + " " + std::to_string(i)) == cosSimil.end())
                    {
                        cosSimil[std::to_string(j) + " " + std::to_string(i)] = cosArray[j];   
                    }
                }
                else if(i == j)
                {
                    cosArray[j] = 0.0;
                }
            } else
            {
               cosArray[j] = 0; 
            }
            if (cosArray[j] > nkArray->getSimil(i,(2*kValue) - 1))
            {
                nkArray->insertElement(i,cosArray[j],j);
            }
        }
        for(int j = 0; j < trainingTranspose->rows; j++)
        {
            cosArray[j] = 0.0;
        }
    }

    kChanged = false;
    nkBuilt = true;

    delete [] lengthArray;
    lengthArray = nullptr;
    delete [] cosArray;
    cosArray = nullptr;
}

void NBRecommender::rebuildNKArray(void)
{
    nkArray = new NKArray(trainingTranspose->rows,kValue);
    for(int i = 0; i < trainingTranspose->rows; i++)
    {

        float curSimil = 0.0;
        for(int j = 0; j < trainingTranspose->rows; j++)
        {
            if (i < j)
            {
                curSimil = cosSimil[std::to_string(i) + " " + std::to_string(j)];
            }
            else if(j < i)
            {
                curSimil = cosSimil[std::to_string(j) + " " + std::to_string(i)];
            }
            else if(i == j)
            {  
                curSimil = 0.0; 
            }
            if (curSimil > nkArray->getSimil(i,(2*kValue) - 1))
            {
                nkArray->insertElement(i,curSimil,j);
            }
        }
    }
    kChanged = false;
    
}

void NBRecommender::insertionSort(float ** array, int left, int right)
{
    for(int i = left; i < right + 1; i++)
    {
        float * temp = array[i];
        int j = i;
        while(j > left && temp[0] > array[j-1][0])
        {
            array[j] = array[j-1];
            j-=1;
        }
        array[j] = temp;
        
    }
}

void NBRecommender::swap(float ** array, int a, int b)
{
    float * temp = array[a];
    array[a] = array[b];
    array[b] = temp;
}

int NBRecommender::pivot(float ** array, int left, int right)
{
    int center = (left + right) / 2;
    
    if(array[center][0] < array[right][0])
    {
        swap(array, right, center);
    }
    if(array[left][0] < array[right][0])
    {
        swap(array, right, left);
    }
    if(array[left][0] < array[center][0])
    {
        swap(array,center,left);
    }
    swap(array,center,(left+1));
    return array[left+1][0];
}

void NBRecommender::partition(float ** array, int left, int right)
{
    if(right - left >= 10000)
    {
        float pivotValue = pivot(array, left, right);
        
        int i = left + 1, j = right;

        while (i < j)
        {
            
            while (pivotValue < array[++i][0])
            {
                
            }
            while (array[--j][0] < pivotValue)
            {
                
            }
            
            if(i < j)
            {
                swap(array,i,j);
            }
        }
        swap(array,j,left+1);
        
        partition(array, left, j-1);
        partition(array,j+1,right);
    }
    else
    {
        insertionSort(array,left,right);
    }
}

void NBRecommender::quickSort(float ** array, int length)
{
    partition(array, 0, length - 1);
}

float ** NBRecommender::pullKTopVals(int userId)
{
    
    int userItemCount = trainingData->rowPtr[userId + 1] - trainingData->rowPtr[userId];
    int * userItems = new int[userItemCount];
    
    float ** kList = new float*[trainingData->rows];
    for(int i = 0; i < userItemCount; i++)
    {
        userItems[i] = trainingData->columnIndex[trainingData->rowPtr[userId] + i];
        std::cout << userItems[i] << std::endl;
        
    }
    
    return kList;
    
    
    
    
}

void NBRecommender::rankKVals(int * itemsArray, int itemsSize,float ** kList,int kListSize)
{
    for(int i = 0; i < kListSize; i++)
    {
        for(int j = 0; j < itemsSize; j++)
        {
            if (kList[i][2] < itemsArray[j])
            {
                kList[i][0] += cosSimil[std::to_string(int(kList[i][2])) + " " + std::to_string(itemsArray[j])];
                
            }
            else if (itemsArray[j] < kList[i][2])
            {
                kList[i][0] += cosSimil[std::to_string(itemsArray[j]) + " " + std::to_string(int (kList[i][2]))];
                
            }
            else if(itemsArray[j] == kList[i][2])
            {
                kList[i][0] += 0.0;
            }
        }
    }
}

float *** NBRecommender::recommendations(void)
{
    
    float *** userRecs =  new float **[trainingData->rows];
    
    if(!nkBuilt)
    {
        buildNKArray();
    }
    
    if(kChanged)
    {
        rebuildNKArray();
    }
    for(int i = 0; i < trainingData->rows; i++)
    {
        
        userRecs[i] = pullKTopVals(i);
        /*
        for(int j = 0; j < nValue; j++)
        {
            std::cout << userRecs[i][j][0];
            std::cout << " ";
            std::cout << userRecs[i][j][1] << std::endl;
        }
        */
    }
    
    return userRecs;
}
//SegFault in testRecs
float * NBRecommender::testRecsHR(float *** userRecs)
{
    float hits = 0.0;
    float aRHR = 0.0;
    
    for(int i = 0; i < testingData->rows; i++)
    {
        
        bool inRec = false;
        int jVal = 0;
        for (int j = 0; j < nValue; j++)
        {
            
            if(testingData->columnIndex[i] == userRecs[i][j][1])
            {
                inRec = true;
                jVal = j + 1;
            }
        }
        if (inRec == true)
        {
            hits += 1.0;
        }
        if (jVal != 0)
        {
            aRHR += 1.0/jVal;
        }
    }
    float * testVals = new float[2];
    testVals[0] = hits/testingData->rows;
    testVals[1] = aRHR/testingData->rows;
    return testVals;
}

void NBRecommender::testingMethod(void)
{
    int kVals[] = {3,5,10,20};
    int nVals[] = {5,10,20};
    
    for(int i = 0; i < 4; i++)
    {
        changeKValue(kVals[i]);
        for(int j = 0; j < 3; j++)
        {
            changeNValue(nVals[j]);
            clock_t start = clock();
            float *** userRecs = recommendations();
            std::cout << (clock() - start)/1000 << std::endl;
            //float * stats = testRecsHR(userRecs);
            std::cout << "recs" << std::endl;
            for(int i = 0; i < trainingData->rows; i++)
            {
                for(int j = 0; j < nValue; j++)
                {
                    delete [] userRecs[i][j];
                    userRecs[i][j] = nullptr;
                }
                delete [] userRecs[i];
                userRecs[i] = nullptr;
            }
            delete [] userRecs;
            userRecs = nullptr;
            //delete [] stats;
        }
    }
}
