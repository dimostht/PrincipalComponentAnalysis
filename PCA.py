import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from ask3 import KMeans,getPurity


# matrix: the matrix we want to reduce
# V: the number of dimensions we want the matrix to be reduced
def pca(matrix,V):
    # find the mean values of the matrix
    mean = np.mean(matrix)
    # compute the centralized matrix
    matrix = matrix - mean
    # find the covarience
    covariance_matrix = np.cov(np.transpose(matrix))
    # find the eigenvectors and eigenvalues
    eigVec,eigVal = np.linalg.eig(covariance_matrix)
    # resize to the wanted number
    eigVal = eigVal[:,:V]
    # sum of last axis of eigenvalues and 2nd last from matrix
    matrix = np.dot(np.transpose(eigVal),np.transpose(matrix))
    return np.transpose(matrix)

# plot the M table with the correct color according to table Ltr for N numbers
# and the Sets after the KMeans
def plot(m,Ltr,n,sets):
    plt.figure()
    
    plt.subplot(2,1,1)
    plt.title("The points with PCA for V = 2")
    # numbers of 0,1,2,3's in N
    count = [0,0,0,0]
    for i in range(len(Ltr)):
        if Ltr[i] == 0:
            count[0]  = count[0] +1
        if Ltr[i] == 1:
            count[1]  = count[1] +1
        if Ltr[i] == 2:
            count[2]  = count[2] +1
        if Ltr[i] == 3:
            count[3]  = count[3] +1

    x = [[] for _ in range(n)]
    y = [[] for _ in range(n)]

    c = 0
    for i in range(len(m)):
        x[c].append(m[i][0])
        y[c].append(m[i][1])
        if i < count[0]:
            c=0
        elif i < count[0] + count[1]:
            c=1
        elif i < count[0] + count[1] + count[2]:
            c=2
        else :
            c=3

    #alpha value
    a = 0.3

    # the 4 colors for the number
    colors = ['r','g','c','y']

    for i in range(n):
        plt.scatter(x[i] , y[i], color=colors[i] , alpha=a)

    plt.grid(True)

    plt.subplot(2,1,2)
    plt.title("The points with K-Means Clustering")
    for i in range(len(sets)):
        x = []
        y = []
        for j in range(len(sets[0])):
            x.append(sets[i][j][0])
            y.append(sets[i][j][1])
        plt.scatter(x , y , color = colors[i] , alpha=a)

    plt.grid(True)
    plt.show()

def main():
    # get the M matrix from the original csv
    m = pd.read_csv("mnist_train.csv")

    #  get the N matrix from the original csv
    nf = pd.read_csv("mnist_test.csv")



    # L train
    Ltr = m.iloc[:, :1].values
    nf.iloc[: , :1]

    # remove the first columns that has the number
    m.drop(m.columns[0], axis=1, inplace=True)
    nf.drop(m.columns[0], axis=1, inplace=True)

    # convert M from data frame to list
    m = m.values
    nf = nf.values

    # number of Numbers in the M and Ltr matrices
    n = 4

    # The M~ Tables for V = 2, 25 ,50 and 100
    mV2 = pca(m, 2).real
    mV25 = pca(m, 25).real
    mV50 = pca(m, 50).real
    mV100 = pca(m, 100).real

    # call the K Means function from ask3 with max iterations = 1000
    maxIterations = 1000

    sets2 = KMeans(n, mV2, maxIterations)

    # plot the M~ 2 table with the color according to the correct value
    # and the Sets after the KMeans
    plot(mV2, Ltr, n, sets2)

    sets25 = KMeans(n, mV25, maxIterations)

    sets50 = KMeans(n, mV50, maxIterations)

    sets100 = KMeans(n, mV100, maxIterations)


    # compute the purities of each set

    purities = []
    labels = [2, 25, 50, 100]

    purities.append(getPurity(mV2, Ltr, sets2, n))
    purities.append(getPurity(mV25, Ltr, sets25, n))
    purities.append(getPurity(mV50, Ltr, sets50, n))
    purities.append(getPurity(mV100, Ltr, sets100, n))

    print(purities)


    # find for which V we have maximum purity
    maxPurity = purities[0]
    maxN = 0

    for i in range(n):
        if purities[i] > maxPurity:
            maxN = i
            maxPurity = purities[i]



    print("Maximum purity is", maxPurity, " for V = ", labels[maxN])

    # compute the N~ table with PCA with the same V as for M
    # we will use this for exercise 5
    nV = pca(nf,labels[maxN]).real
    # save as DataFrame
    dfNV = pd.DataFrame(data=nV)

    # we save the PCA file with the maximum purity to a file called MV.csv

    if maxN == 0:
        dfMV = pd.DataFrame(data=mV2)
    if maxN == 1:
        dfMV = pd.DataFrame(data=mV25)
    if maxN == 2:
        dfMV = pd.DataFrame(data=mV50)
    if maxN == 3:
        dfMV = pd.DataFrame(data=mV100)

    # store the M~ and N~ tables to be used in exercise 5
    dfMV.to_csv(r'Files/MV.csv', index=False)
    dfNV.to_csv(r'Files/NV.csv', index=False)




