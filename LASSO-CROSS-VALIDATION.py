import math; import random; import matplotlib.pyplot as plt; import sympy as sp; import scipy.optimize; import numpy as np
from sympy.utilities.lambdify import lambdify; import sys

sys.getdefaultencoding()

### COMMENTS
    # I can probably make the chi-square formula as a function to avoid writing it everytime. 

#FIRST >>> LASSO
    #DATA GENERATION PART

def y(x):
    return 2+(0.5)*x-(0.1)*(x-5)**2

max=14.25 # Data generation will be yield from x=0 to x=max
delta=0.25 # This is the distance between points
sigma=0.12 # This is the data uncertainty

raerr = []; datnoerr = np.array([[0,0]],dtype=np.float64); data = np.array([[0,0]],dtype=np.float64); x_data = []; y_data = []

for i in range(math.floor(max/delta+4)):
    raerr.append(random.normalvariate(0,sigma))

row_i = 0
for i in np.arange(0,max,delta):
    datnoerr[row_i][0] = i 
    datnoerr[row_i][1] = y(i)
    if(i!=max-delta):
        datnoerr = np.vstack((datnoerr,[0,0])) #What does axis mean?
    row_i += 1

### print(datnoerr)

le = len(datnoerr)

### print(le)

row_i = 0
for i in range(le):
    data[row_i][0] = datnoerr[i][0]
    data[row_i][1] = datnoerr[i][1]+raerr[i]
    if(i!=le-1):
        data = np.vstack((data,[0,0]))
    row_i += 1

for i in range(le):
    x_data.append(data[i][0])
    y_data.append(data[i][1])

plt.errorbar(x_data,y_data,yerr=sigma,ecolor='#4B8BC8',color='#4B8BC8',linestyle='',fmt='.',capsize=2.25)
plt.title(str(le) + ' data points with \u03C3 = .12') 
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid(True)
plt.show()

mmax = 7
a = sp.symbols('a0:%d'%mmax) #Good generalization

def h(x):
    SUMMATION = 0
    for i in range(mmax):
        SUMMATION += a[i]*x**i
    return SUMMATION

    #FIRST EXERCISE; GET CHI-SQUARE AND D.O.F

chisq = 0
for i in range(0,le):
    chisq += ((h(data[i][0])-data[i][1])**2)/((sigma)**2) #Definition of chi-square.

chisq_dummy = lambdify(a,chisq)

def chisq_v(a):
    return chisq_dummy(*tuple(a))

temp = scipy.optimize.minimize(chisq_v,np.zeros(mmax),method='Nelder-Mead',options={'maxiter':2500,'disp':True}) #Why is this method working and the other doesn't?
chisq_res = temp.fun
chisqparams = temp.x

X = np.arange(0, max, 0.05)
def H(x):
    SUMMATION = 0
    for i in range(mmax):
        SUMMATION += chisqparams[i]*x**i
    return SUMMATION

plt.plot(X,H(X),color='#2E6598')
plt.errorbar(x_data,y_data,yerr=sigma,ecolor='#4B8BC8',color='#4B8BC8',linestyle='',fmt='.',capsize=2.25)
plt.title(str(le) + ' data points with \u03C3 = .12')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid(True)
plt.show()

print("The chi-square is equal to: " + str(chisq_res))
print("The degrees of freedom are equal to: " + str((chisq_res/(le-(mmax+1)))))

    #SPLITTING DATA

training =  np.array([[0,0]],dtype=np.float64) 
test = np.array([[0,0]],dtype=np.float64)
sptrain_n = math.ceil(le*(.7))
sptest_n = le - sptrain_n
randmore = []

for i in range(le):
    randmore.append(i)

row_i=0
while(len(randmore) > sptest_n):

    r = random.randint(0,le) #GENERATION OF A RANDOM INTEGER WITH UNIFORM DISTRIBUTION

    for i in range(len(randmore)):
        if(randmore[i]==r):
            training[row_i][0] = data[r][0] 
            training[row_i][1] = data[r][1]
            
            if(len(randmore) != sptest_n+1):
                training = np.vstack((training,[0,0]))

            randmore.pop(i)

            row_i += 1
            break

row_i=0
for i in range(sptest_n):
    test[row_i][0] = data[randmore[i]][0]
    test[row_i][1] = data[randmore[i]][1]
    
    if(row_i < sptest_n-1):
        test = np.vstack((test,[0,0]))
        
    row_i+=1

    #CALCULATING CHI OF TRAINING AND VALIDATION SET

chisq_training = 0
for i in range(sptrain_n):
    chisq_training += (h(training[i][0])-training[i][1])**2/(sigma)**2 
chisq_dummy_training = lambdify(a,chisq_training)

def chisq_v_training(a):
    return chisq_dummy_training(*tuple(a))

temp_training = scipy.optimize.minimize(chisq_v_training,np.array([1,1,1,1,0,0,0]),method='Nelder-Mead',options={'maxiter':2500,'disp':True}) 
chisq_training_res = temp_training.fun
chisq_training_params = temp_training.x

def fitted(x):
    SUMMATION = 0
    for i in range(mmax):
        SUMMATION += a[i]*x**i

    SUMMATION = SUMMATION.subs(list(zip(a,chisq_training_params)))

    return SUMMATION

chisq_val = 0
for i in range(sptest_n):
    chisq_val += (fitted(test[i][0])-test[i][1])**2/(sigma)**2 

X = np.arange(data[0][0], data[le-1][0]+.25, 0.05)

X_Y = []
for i in X:
    X_Y.append(fitted(i))

plt.plot(X,X_Y,color='#2E6598') 
plt.plot(training[:,0],training[:,1],color='y',linestyle='',label='Training',marker='.')
plt.plot(test[:,0],test[:,1],color='b',linestyle='',label='Validation',marker='.')
plt.title(str(le) + ' data points with \u03C3 = .12')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid(True)
plt.legend()
plt.show()

print('\n')
print('The training chi square is: ' + str(chisq_training_res))
print('The validation chi square is: ' + str(chisq_val))


#SECOND >>> CROSS VALIDATION
    #SET-UP (CHI SQUARE OF THE TRAINING AND VALIDATION SET WITH THE PENALTY TERM ADDED)

penalty = 0
Lambda = sp.Symbol('Lambda')
for i in range(mmax):
    penalty += abs(a[i])
penalty *= Lambda**4

chisq_training_crossv = 0
for i in range(sptrain_n):
    chisq_training_crossv += (h(training[i][0])-training[i][1])**2/(sigma)**2 
chisq_training_crossv += penalty

coefficients = [sp.Symbol(f'a{i}') for i in range(7)]
lambdish_cv_dummy = lambdify((coefficients,Lambda),chisq_training_crossv) 

unique = set()
for i in sp.preorder_traversal(chisq_training_crossv):
    if isinstance(i,sp.Symbol):
        unique.add(i)
count = len(unique)

def lambdish_cv(a):
    b = [a[i] for i in range(count-1)]
    return lambdish_cv_dummy(b,a[count-1])

chiq_training_lasso = scipy.optimize.minimize(lambdish_cv,np.zeros(count),method='Nelder-Mead',options={'maxiter':2500,'disp':True}) 
best_params = temp_training.x

print(best_params)

def fitted_lasso(x):
    SUMMATION = 0
    for i in range(mmax):
        SUMMATION += best_params[i]*x**i
    return SUMMATION

CHI_CHI = []

chisq_v_lasso = 0
for i in range(sptest_n):
    chisq_v_lasso += (fitted_lasso(test[i][0])-test[i][1])**2/(sigma)**2 
    CHI_CHI.append(chisq_v_lasso)

LAMBDA_val = []
for i in np.arange(0,8.5,.5):
    LAMBDA_val.append(i)

print(LAMBDA_val)
print(CHI_CHI)

plt.plot(LAMBDA_val,CHI_CHI)
plt.yscale('log')
plt.show()

    #EXERCISE 1 ::: ONE-SIGMA RULE (USED FOR THE PENALTY TERM)

kf = 5
rlist = []
for i in range(kf):
    rlist.append(random.randint(6))

    #EXERCISE 2 ::: VARIATION OF THE PENALTY TERM



    #EXERCISE 3 ::: AIC, AICc and BIC
