import numpy as np
import matplotlib.pyplot as plt
import math
import random


def rng(matrix_of_zeros, top_value_of_matrix, remaining_values):
    position = random.randrange(0, remaining_values, 1)
    position_state = 0
    value_of_position = 0
    for SearchOfZeros in range(0, top_value_of_matrix, 1):
        if matrix_of_zeros[SearchOfZeros] == 0:
            position_state += 1
            value_of_position = SearchOfZeros

        if position == (position_state - 1):
            return value_of_position


def EuclideanDistance(clu, cent):
    return math.sqrt(
        ((clu[0] - cent[0]) ** 2) + ((clu[1] - cent[1]) ** 2) + ((clu[2] - cent[2]) ** 2) + ((clu[3] - cent[3]) ** 2))


dataset = np.loadtxt('Mall_Customers.txt', dtype=int)

# Number of Customers in data set
NumberofSamples = 200

customerset = np.zeros((NumberofSamples, 4), dtype=int)
for i in range(0, 200, 1):
    customerset[i][0] = dataset[i][1]
    customerset[i][1] = dataset[i][2]
    customerset[i][2] = dataset[i][3]
    customerset[i][3] = dataset[i][4]



# Number of times the centroids will be updated
UpdateTimes = 10

# Updating position of centroid
UpdationgPositionOfCentroid = 40

GridSizeGender = 5
GridSizeAge = 100
GridSizeIncome = 100
GridSizeSpendingScore = 100

# Number of Clusters
Ncentroid = 5

# defining first value as big as possible respect to grid size
JPrime = 2 * NumberofSamples
clusterPrime = np.zeros((Ncentroid, 4), dtype=float)
groupsPrime = np.zeros((NumberofSamples, Ncentroid), dtype=int)

#########
for updationgPositionOfCentroid in range(0, UpdationgPositionOfCentroid, 1):

    # Centroids positions are defined as overlapped to the particle points
    Pcluster = np.zeros(NumberofSamples, dtype=int)
    cluster = np.zeros((Ncentroid, 4), dtype=float)
    for i in range(0, Ncentroid, 1):
        PositionOfCentroid = rng(Pcluster, NumberofSamples, NumberofSamples - i)
        Pcluster[PositionOfCentroid] = 1
        cluster[i][0] = customerset[PositionOfCentroid][0]
        cluster[i][1] = customerset[PositionOfCentroid][1]
        cluster[i][2] = customerset[PositionOfCentroid][2]
        cluster[i][3] = customerset[PositionOfCentroid][3]

    JNew = 2 * NumberofSamples
    clusterNew = np.zeros((Ncentroid, 4), dtype=float)
    groupsNew = np.zeros((NumberofSamples, Ncentroid), dtype=float)

    for UpdateCentroids in range(0, UpdateTimes, 1):

        euclideanDistance = np.zeros((NumberofSamples, Ncentroid), dtype=float)
        for i in range(0, NumberofSamples, 1):
            for j in range(0, Ncentroid, 1):
                euclideanDistance[i, j] = EuclideanDistance(customerset[i], cluster[j])

        # creating cluster groups
        groups = np.zeros((NumberofSamples, Ncentroid), dtype=float)

        # counts number of points in the cluster
        Ningroup = np.zeros((Ncentroid, 1), dtype=float)

        # Sum of the group
        Singroup = np.zeros((Ncentroid, 4), dtype=float)

        # optimization parameter
        J = 0

        for i in range(0, NumberofSamples, 1):
            smallest = NumberofSamples * 2
            for j in range(0, Ncentroid, 1):
                if smallest > euclideanDistance[i][j]:
                    smallest = euclideanDistance[i][j]

            for j in range(0, Ncentroid, 1):
                if smallest == euclideanDistance[i][j]:
                    groups[i][j] = 1
                    Ningroup[j] += 1
                    J += smallest

                    for k in range(0, 4, 1):
                        Singroup[j][k] += customerset[i][k]

        # mean of each cluster
        Mingroup = np.zeros((Ncentroid, 4), dtype=float)
        Mingroup = Singroup / Ningroup

        J *= (1 / NumberofSamples)

        print('J=', J)
        print('JNew = ', JNew)

        print(updationgPositionOfCentroid, '.th main loop...', UpdateCentroids + 1, '.th Loop ended here')
        print('...........................')
        if JNew > J:
            JNew = J
            clusterNew = Mingroup
            cluster = Mingroup
            groupsNew = groups

        print('Jprime = ', JPrime)

    if JPrime > JNew:
        JPrime = JNew
        clusterPrime = clusterNew
        groupsPrime = groupsNew

    print('Jprime = ', JPrime)

    print('Centroid position updater', updationgPositionOfCentroid, ' Times')

# here show the grouped data point with different color and shape
g = np.zeros((NumberofSamples, 1), dtype=int)
for i in range(0, NumberofSamples, 1):
    for j in range(0, Ncentroid, 1):
        if groupsPrime[i][j] != 0:
            g[i] = j
colors = ("red", "green", "blue", "gray", "yellow", )
m = ('.', ',', 'o', 'v', '^')
plt.grid()
plt.title('Costumer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
for i in range(0, NumberofSamples, 1):
    plt.scatter(customerset[i][2], customerset[i][3], c=colors[g[i]], marker=m[g[i]])
for i in range(0, Ncentroid, 1):
    plt.scatter(clusterPrime[i][2], clusterPrime[i][3], s=40, c='magenta')
plt.show()