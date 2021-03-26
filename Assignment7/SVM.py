import numpy as np
from numpy import random
from matplotlib import pyplot as plt


class SVM():

    def __init__(self, v1, v2, epsilon, iteration=1000, minChange=0.001, kernel_stdev=None, maxPass=10):
        if 1/v1 < 1:
            raise Exception('1/v1>=1')

        if 1/v2 < 1:
            raise Exception('1/v2>=1')

        self.epsilon = epsilon
        self.kernel_stdev = kernel_stdev
        self.v1 = v1
        self.v2 = v2
        self.high = 0
        self.highPrime = 0
        self.p1 = 0
        self.p2 = 0
        self.iter = iteration
        self.maxPass = max(maxPass, (int)(0.1*iteration))
        self.K = np.zeros(1)
        self.gamma = np.zeros(1)
        self.X = np.zeros(1)
        self.dotMem = np.zeros(1)
        self.minChange = minChange

    def kernel(self, x1, x2):
        if self.kernel_stdev is None:
            return np.dot(x1.T, x2)
        else:
            return np.exp((-1/2.0)*((np.linalg.norm(x1-x2)/self.kernel_stdev)**2))

    # if linear kernel is used
    def getW(self):
        if self.kernel_stdev is not None:
            print("Warning kernel is not linear ")

        return np.dot(self.X.T, self.gamma)

    def calcGamma(self, a, b, tstar):
        mu = (self.K[a][a]+self.K[b][b]-2*self.K[a][b])
        val = self.dotMem[a]-self.dotMem[b]
        # fd = tstar*(self.K[a][a]-self.K[a][b]) - \
        self.gamma[b] * mu + val
        #dd = -1*mu

        # print((self.dotMem[a]-self.dotMem[b])/mu,'change')
        # print(self.gamma[b],1)
        self.gamma[b] = self.gamma[b] + val/mu

    def predict(self, X):
        m2 = X.shape[0]
        ans = np.zeros((m2))

        for i in range(m2):
            val = 0
            for j in range(self.X.shape[0]):
                val += self.gamma[j]*self.kernel(self.X[j], X[i])

            if np.sign((val-self.p1))*np.sign((self.p2-val)) >= 0:
                ans[i] = 1
            else:
                ans[i] = 0

        return ans

    def calcKernel(self, X, m):
        self.K = np.zeros((m, m))

        for i in range(m):
            for j in range(m):
                self.K[i][j] = self.kernel(X[j], X[i])

    def minDistanceFromHyperplanes(self, b):
        val = self.dotMem[b]
        return min(val-self.p1, self.p2-val)

    def getTstar(self, a, b):
        return self.gamma[a]+self.gamma[b]

    def adjustP(self):
        pos = np.where((self.gamma > 0) & (self.gamma < self.high))
        n1 = len(pos[0])
        if n1 == 0:
            return False
        p1 = np.sum(self.dotMem[pos])

        pos = np.where((self.gamma < 0) & (self.gamma > -1*self.highPrime))
        n2 = len(pos[0])
        if n2 == 0:
            return False
        p2 = np.sum(self.dotMem[pos])

        # if n1 > 0:
        #     self.p1 = p1/n1
        # else:
        #     # this is a very big issue
        #     self.p1 = self.p1  # 0
        # if n2 > 0:
        #     self.p2 = p2/n2
        # else:
        #     # and this as well
        #     self.p2 = self.p2  # 0

        self.p1 = p1/n1
        self.p2 = p2/n2

        return True

    def boundGamma(self, tstar, b):
        high = min(self.high, tstar+self.highPrime)
        low = max(tstar-self.high, -1*self.highPrime)

        self.gamma[b] = min(self.gamma[b], high)
        # print(self.gamma[b],2)
        self.gamma[b] = max(self.gamma[b], low)
        # print(self.gamma[b],3)

    def isVoilating(self, b):
        val = np.sign((self.dotMem[b]-self.p1)) * \
            np.sign((self.p2-self.dotMem[b]))

        g = self.gamma[b]

        if g == 0:
            if val > 0:
                return False

        elif g > -1*self.highPrime and g < 0:
            if val == 0:
                return False

        elif g == -1*self.highPrime:
            if val < 0:
                return False

        elif g > 0 and g < self.high:
            if val == 0:
                return False

        elif g == self.high:
            if val < 0:
                return False

        return True

    def takeStep(self, a, b):
        orgGamma = self.gamma[b]

        tstar = self.getTstar(a, b)
        self.calcGamma(a, b, tstar)
        self.boundGamma(tstar, b)

        #print(abs((orgGamma-self.gamma[b])/(orgGamma + 1e-9)))
        if abs((orgGamma-self.gamma[b])/(orgGamma + 1e-9)) < self.minChange:
            self.gamma[b] = orgGamma
            return False

        self.gamma[a] = tstar-self.gamma[b]
        return True

    def examine(self, b):
        bval = self.minDistanceFromHyperplanes(b)
        m = len(self.gamma)

        arr = np.zeros((m, 2))

        for i in range(m):
            arr[i][0] = i
            arr[i][1] = -1*abs(bval-self.minDistanceFromHyperplanes(i))

        arr = arr[arr[:, 1].argsort()]

        for v in arr:
            if v[0] == b:
                continue

            # print(v,b)
            gammaB = self.gamma[b]
            gammaA = self.gamma[(int)(v[0])]

            if self.takeStep((int)(v[0]), b):
                self.dotMem = np.dot(self.K, self.gamma)
                if not self.adjustP():
                    self.gamma[b] = gammaB
                    self.gamma[(int)(v[0])] = gammaA
                    continue
                # print('true')
                return True

            # print('false')

        return False

    def fit(self, X):
        m, _ = X.shape

        self.X = X

        self.high = 1/(self.v1*m)
        self.highPrime = self.epsilon/(self.v2*m)

        self.gamma = np.random.rand(
            m)*(self.high + self.highPrime)-self.highPrime

        print(np.min(self.gamma), np.max(self.gamma))

        self.calcKernel(X, m)
        self.dotMem = np.dot(self.K, self.gamma)

        self.p2 = np.max(self.dotMem)
        self.p1 = np.min(self.dotMem)
        # self.adjustP()

        if self.p1 == 0:
            self.p1 = random.uniform(-m, -m/2)

        if self.p2 == 0:
            self.p2 = random.uniform(m/2, m)

        # return

        w = np.array([0, 0])

        if self.kernel_stdev is None:
            w = self.getW()

        print('On start')

        if w[0] and w[1]:
            plt.figure(figsize=(8, 8))
            plt.scatter(X[:, 0], X[:, 1])
            plt.axline(((self.p1-w[1])/w[0], 1),
                       (1, (self.p1-w[0])/w[1]), c='r')
            plt.axline(((self.p2-w[1])/w[0], 1),
                       (1, (self.p2-w[0])/w[1]), c='g')
            plt.show()

        KKTviolators = 0
        for i in range(m):
            if self.isVoilating(i):
                KKTviolators += 1

        k = {'KKT voilators count': KKTviolators,
             'Maximisation Function Value': -1/2*np.dot(self.dotMem, self.gamma)}
        print(k)

        handleAll = 1
        passes = 0

        for _ in range(self.iter):
            b = -1

            if handleAll:
                b = random.randint(0, m-1)
            else:
                mval = -1

                for i in range(m):
                    val = abs(self.minDistanceFromHyperplanes(i))
                    if val > mval:
                        mval = val
                        b = i

            flag = self.examine(b)

            if flag:
                passes = 0
            else:
                passes = passes + 1
                if passes > self.maxPass:
                    break

            if handleAll:
                handleAll = 0
            elif flag == False:
                handleAll = 1

            # print(-1/2*np.dot(self.dotMem,self.gamma))

            KKTviolators = 0

            # for i in range(m):
            #     if self.isVoilating(i):
            #         KKTviolators += 1

            # print(KKTviolators)

            for i in range(m):
                if self.isVoilating(i):
                    KKTviolators += 1
                    if KKTviolators >= 2:
                        break

            if KKTviolators < 2:
                break

        print('At end')

        if self.kernel_stdev is None:
            w = self.getW()

        if w[0] and w[1]:
            plt.figure(figsize=(8, 8))
            plt.scatter(X[:, 0], X[:, 1])
            plt.axline(((self.p1-w[1])/w[0], 1),
                       (1, (self.p1-w[0])/w[1]), c='r')
            plt.axline(((self.p2-w[1])/w[0], 1),
                       (1, (self.p2-w[0])/w[1]), c='g')
            plt.show()

        KKTviolators = 0
        for i in range(m):
            if self.isVoilating(i):
                KKTviolators += 1

        k = {'KKT voilators count': KKTviolators,
             'Maximisation Function Value': -1/2*np.dot(self.dotMem, self.gamma)}
        print(k)

        return k
