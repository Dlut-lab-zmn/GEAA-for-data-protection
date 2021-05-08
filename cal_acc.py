import math
def acc(p_,r_,acc_):
    acc_1 = (acc_ - p_)/acc_
    acc_2 = math.pow( 2, math.pow(r_/acc_,2) ) - 1
    return acc_1*acc_2

print(   (acc(60.78,73.28,88.89)+acc(45.31,72.33,86.78)+acc(51.81,69.63,86.34)+acc(59.22,74.66,92.12)) / 4.)