# 2018.8.24 by Jason Wang
# reference https://blog.csdn.net/pengwill97/article/details/77200372

import numpy as np

# store the solution
x= []

# maximum common divisor
def gcd(a,b):
    if b == 0:
        return a
    else:
        return gcd(b,a%b)

# least common multiple
def lcm(a,b):
    return a/gcd(a,b)*b

#Gauss-Jordan elimination -2:float solution, no integer solution -1: no solution 0:unique solution >0:infinite solution

def Gauss(a,equ,var):
    i = 0
    j = 0
    k = 0


# convert to step matrix
    col = 0
    for k in range(equ):
        if col < var :
            max_r = k
            # get the maximum row index in current column
            for i in range(k+1,equ):
                if (abs(a[i][col])>abs(a[max_r][col])):
                    max_r = i

            #swap the maximum row into the first row
            if max_r != k:
                for j in range(k,var+1):
                    temp = a[k][j]
                    a[k][j] = a[max_r][j]
                    a[max_r][j] = temp

            # if the value is zero, we should shift right, not right-bottom
            if a[k][col] == 0:
                k = k-1
                continue

            # eliminate the value in this column to ensure the nonzero value only in the first row
            for i in range(k+1,equ):
                if a[i][col] != 0:
                    LCM = lcm(abs(a[k][col]),abs(a[i][col]))
                    ta = LCM/abs(a[i][col])
                    tb = LCM/abs(a[k][col])
                    # if the symbol is opposite, we need to add instead of minus
                    if a[i][col] * a[k][col] < 0 :
                        tb = -tb
                    for j in range(col,var+1):
                        a[i][j] = a[i][j]*ta - a[k][j]*tb
            col = col + 1

    #different from the C++ code, in C++ code the for loop will add k finally,but python need to add manually.
    k = k + 1

    #this circumstance that coefficient is zero but the result is nonzero
    for i in range(k,equ):
        if a[i][col] != 0:
            return -1

    #this circumstance that we have more valuable that not unique
    if k < var:
        return var - k

    #this circumstance that we have unique solution
    for i in range(var-1,-1,-1):
        temp = a[i][var]
        # we can get the solution from the bottom to up
        for j in range(i+1,var):
            if a[i][j]!=0:
                temp = temp - a[i][j]*x[var-j-1]
        if temp % a[i][i] !=0:
            return -2
        x.append(temp / a[i][i])
    return 0

if __name__ == '__main__':
    a = [[1,2,3,-6],
         [3,2,2,7],
         [4,1,1,6]]
    print(Gauss(a,3,3))
    for i in range(len(x)):
        print(x[len(x)-i-1])


