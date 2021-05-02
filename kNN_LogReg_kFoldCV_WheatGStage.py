#import libraries
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from math import floor
#from random import randint
from decimal import Decimal
from datetime import datetime
import random
import csv
import sklearn
from PIL import Image
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from random import randrange
from random import seed
from statistics import mean 
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

def xyMatadata():
    now1 = datetime.now()
    starting_time = now1.strftime("%H:%M:%S")
    timestamp1 = datetime.timestamp(now1)
    
    # load the dataset from the CSV file
    file_dir = "C:/Users/CONALDES/Documents/WheatGrouthStage/Images/"       

    print("### Input Data ###")
    print("==================")
        
    img_uids_list = []     
    files = os.listdir(file_dir)
    for f in files:
        file_name = file_dir + f        
        if os.path.isfile(file_name):
            if file_name[-3:] == "bmp":
                img_uids_list.append([file_name[-12:-4], f])
            elif file_name[-3:] == "png":
                img_uids_list.append([file_name[-12:-4], f])
            elif file_name[-3:] == "jpg":
                img_uids_list.append([file_name[-12:-4], f])
            elif file_name[-4:] == "jpeg":
                img_uids_list.append([file_name[-13:-5], f])                
       
    # load the dataset from the CSV file 
    reader = csv.reader(open("C:/Users/CONALDES/Documents/WheatGrouthStage/Train.csv", "r"), delimiter=",")
    xx = list(reader)
    xxln = len(xx)
    allrecs = []
    for row in range(1, xxln):
        fields = []
        recln = len(xx[row])
        for i in range(0, recln):
            fields.append(xx[row][i])    
        allrecs.append(fields)

    features = np.array(allrecs)

    UIDtrain = features[:,0]
    print("UIDtrain: " + str(UIDtrain))
    ytrain = features[:,1].astype("float")
    print("ytrain.shape: " + str(ytrain.shape))
    gstlqtyln = len(ytrain)

    train_xylist = []
    train_xlist = []
    y_data = []
    test_ID = []
    test_xlist = []    
    
    uidsln = len(UIDtrain)
    imguidslstln = len(img_uids_list)
    for i in range(0, imguidslstln):
        seen = False
        for j in range(0, uidsln):
            if img_uids_list[i][0] == UIDtrain[j]:
                train_xylist.append([img_uids_list[i][1], ytrain[j]])
                #y_data.append(ytrain[j])                     
                seen = True
                break

        if seen == False:                       
            test_ID.append(img_uids_list[i][0])
            test_xlist.append(img_uids_list[i][1])
            
    #random.shuffle(train_xylist)
    #train_xylist = train_xylist[:2193]  # Exactly 2193 elements
    #train_xylist = train_xylist[:193]
    #train_xylist = train_xylist[:73]
    trainxyln = len(train_xylist)
    for i in range(0, trainxyln):
        train_xlist.append(train_xylist[i][0])
        y_data.append(train_xylist[i][1])

    print("len of img_uids_list: " + str(len(img_uids_list)))
    print("                            ")
    print("len of train_xlist: " + str(len(train_xlist)))
    print("                            ")
    print("len of test_xlist: " + str(len(test_xlist)))
    print("                            ")
    print("Image conversion to grayscale going on ......")
    print("                            ") 
    print("Image conversion to grayscale for train images")   
    imgLstLen = len(train_xlist)
    #x, invaltrain_imgs = getAgeRGBArrays3(file_dir, train_xlist)
    x, invaltrain_imgs = getAgeRGBArrays9(file_dir, train_xlist)
    print("                            ") 
    print("Image conversion to grayscale for test images")  
    #x_test, invaltest_imgs = getAgeRGBArrays3(file_dir, test_xlist)
    x_test, invaltest_imgs = getAgeRGBArrays9(file_dir, test_xlist)
    #x = x.astype("float")
    #x_test = x_test.astype("float")
    y = []
    invtrimgln = len(invaltrain_imgs)
    ydataln = len(y_data)

    for j in range(0, ydataln):
        seen = False
        for k in range(0, invtrimgln):        
            if invaltrain_imgs[k] == j:
                seen = True
                break
            
        if seen == False:         
            y.append(y_data[j])
                
          
    print("                              ")
    print("x: " + str(x))
    print("                              ")
    print("x_test: " + str(x_test))
    print("                              ")
    print("y: " + str(y))
    print("                              ")

    print("len(x): " + str(len(x)))
    print("len(x_test): " + str(len(x_test)))
    print("len(y): " + str(len(y)))

    # we set a threshold at 80% of the data
    '''
    m = len(x)
    m_train_set = int(m * 0.8)
        
    print("### Traning Set (80%) and Testing Set (20%) ###")
    print("===============================================")
    print("m_train_set: " + str(roundup(m_train_set,0)))
    print("m_test_set: " + str(roundup((m - m_train_set),0)))                   
    print("                              ")

    # we split the train and test set using the threshold    
    X_train, X_val = x[:m_train_set], x[m_train_set:]
    Y_train, y_val = y[:m_train_set], y[m_train_set:]

    print("                              ")
    print("X_train: " + str(X_train))
    print("Y_train: " + str(Y_train))                   
    print("                              ")    
    print("X_val: " + str(X_val))
    print("y_val: " + str(y_val))                       
    print("                              ")
    '''
    

    '''
    print("                              ")
    print("### Normalisation process going on .......... ###")
    print("                              ")

    
    # version 1
    mean_Col = [np.mean(x[:,c]) for c in range(xcols)] 
    std_Col = [np.std(x[:,c]) for c in range(xcols)]
    print("mean_Col: " + str(mean_Col))
    print("std_Col: " + str(std_Col))

    miny = y.min()
    maxy = y.max()
    print("miny: " + str(miny))
    print("maxy: " + str(maxy))
    
    yrows, ycols = y.shape
    for row in range(0, yrows):
        y[row][0] = (y[row][0] - miny)/(maxy - miny)
    
    for j in range(0, xcols):      
        for row in range(xrows):
            #print('j, row, maxCol, minCol, temp: ' +  str(j) + ', ' + str(row) + ', ' + str(minCol[j]) + ', ' + str(maxCol[j]))
            temp = (x[row][j] - mean_Col[j])/std_Col[j]
            x[row][j] = temp

    for j in range(0, testxcols):    
        #tempval = x[:,j]            
        #meanCol[j] = np.mean(tempval)
        #stdCol[j] = np.std(tempval)        
        for row in range(testxrows):            
            temp = (test_x[row][j] - mean_Col[j])/std_Col[j]
            test_x[row][j] = temp
    

    # version 2
    mean_RGB = [np.mean(x[:,c]) for c in range(xcols)] 
    std_RGB = [np.std(x[:,c]) for c in range(xcols)]
    print("mean_RGB: " + str(mean_RGB))
    print("std_RGB: " + str(std_RGB))
    
    yrows, ycols = y.shape
    miny = [y[:,c].min() for c in range(ycols)]
    maxy = [y[:,c].max() for c in range(ycols)]
    print("miny: " + str(miny))
    print("maxy: " + str(maxy))

    for j in range(0, ycols): 
        for row in range(0, yrows):
            y[row][j] = (y[row][j] - miny[j])/(maxy[j] - miny[j])       
    
    for j in range(0, xcols):      
        for row in range(xrows):
            #print('j, row, maxCol, minCol, temp: ' +  str(j) + ', ' + str(row) + ', ' + str(minCol[j]) + ', ' + str(maxCol[j]))
            temp = (x[row][j] - mean_RGB[j])/std_RGB[j]
            x[row][j] = temp

    
    for j in range(0, testxcols):    
        #tempval = x[:,j]            
        #meanCol[j] = np.mean(tempval)
        #stdCol[j] = np.std(tempval)        
        for row in range(testxrows):            
            temp = (test_x[row][j] - mean_RGB[j])/std_RGB[j]
            test_x[row][j] = temp
    '''
    x = np.vstack(x)
    y = np.vstack(y)
    xy_train = np.concatenate((x, y), axis=1)

    '''
    with open("C:/Users/CONALDES/Documents/WheatGrouthStage/xy_train.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['SumR', 'SumG', 'SumB', 'GrowthStage']) 
        for row in xy_train:    
            l = list(row)    
            writer.writerow(l)
    '''
    
    with open("C:/Users/CONALDES/Documents/WheatGrouthStage/xy_train9.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['SumR1', 'SumG1', 'SumB1', 'SumR2', 'SumG2', 'SumB2', 'SumR3', 'SumG3', 'SumB3', 'GrowthStage']) 
        for row in xy_train:    
            l = list(row)    
            writer.writerow(l)
            
    ID_test = np.vstack(test_ID)
    x_test = np.vstack(x_test)
    Idx_test = np.concatenate((ID_test, x_test), axis=1)

    '''
    with open("C:/Users/CONALDES/Documents/WheatGrouthStage/Idx_test.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['UID', 'SumR', 'SumG', 'SumB']) 
        for row in Idx_test:    
            l = list(row)    
            writer.writerow(l)
    '''
    
    with open("C:/Users/CONALDES/Documents/WheatGrouthStage/Idx_test9.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['UID', 'SumR1', 'SumG1', 'SumB1', 'SumR2', 'SumG2', 'SumB2', 'SumR3', 'SumG3', 'SumB3']) 
        for row in Idx_test:    
            l = list(row)    
            writer.writerow(l)
    print("                              ")
    print("x: " + str(x))
    print("x_test: " + str(x_test))
    print("y: " + str(y))
    print("                              ")
    
    #metadata = {'datax':data_x, 'datay':data_y, 'testUIDs':test_UIDs,'meanRgb':meanRgb,'stdRgb':stdRgb,'mincol':min_col,'maxcol':max_col}
    #return metadata
    
    #return x, y, test_x, test_UIDs, xrows, xcols, ycols, mean_RGB, std_RGB, miny, maxy
    #return test_ID, x, y, x_test

def getAgeRGBArrays3(file_dir, img_list):
    pxel_array = []
    img_list_len = len(img_list)
    invalid_imgs = []
    for i in range(0, img_list_len):
        try:
            file_name = file_dir + img_list[i]
            print("rec no, file_name: " + str(i) + ", " + str(file_name[-13:]))
            img = Image.open(file_name, "r")
            pix_val = list(img.getdata())
            #pix_val_flat = [x for sets in pix_val for x in sets]
            #print("pix_val: " + str(pix_val))
            sum_elem0 = 0
            sum_elem1 = 0
            sum_elem2 = 0
        
            listlen = len(pix_val)
            for l in range(0, listlen):
                sum_elem0 += pix_val[l][0]
                sum_elem1 += pix_val[l][1]
                sum_elem2 += pix_val[l][2]

            
            sum_elem0 = float(sum_elem0)
            sum_elem1 = float(sum_elem1)
            sum_elem2 = float(sum_elem2)
            
            pxel_array.append([sum_elem0,sum_elem1,sum_elem2])
        except TypeError as err:
            print('Handling run-time error:', err)
            invalid_imgs.append(i)                

    #pxelArray = np.vstack(pxel_array)    
    #feature_set = np.vstack(pxel_array)
    #print("pxelArray.shape: " + str(pxelArray.shape))
    #print("feature_set.shape: " + str(feature_set.shape))
    #print("pxelArray: " + str(pxelArray))
    #print("feature_set: " + str(feature_set))
    #return pxelArray, invalid_imgs
    return pxel_array, invalid_imgs

def getAgeRGBArrays9(file_dir, img_list):
    pxel_array = []
    img_list_len = len(img_list)
    invalid_imgs = []
    for i in range(0, img_list_len):
        try:
            file_name = file_dir + img_list[i]
            print("rec no, file_name: " + str(i) + ", " + str(file_name[-13:]))
            img = Image.open(file_name, "r")
            pix_val = list(img.getdata())                       
            img_array1 = []
            img_array2 = []
            img_array3 = []        
            temp_array = []        
            listlen = len(pix_val)
            n1rd = int(listlen/3)
            n2rd = int((listlen - n1rd)/2)
            n3rd = listlen - n1rd - n2rd
            trd = n1rd + n2rd + n3rd
            sum_elem0 = 0
            sum_elem1 = 0
            sum_elem2 = 0 
            for l in range(0, n1rd):
                sum_elem0 += pix_val[l][0]
                sum_elem1 += pix_val[l][1]
                sum_elem2 += pix_val[l][2]
                        
            img_array1.append([sum_elem0,sum_elem1,sum_elem2])            

            sum_elem0 = 0
            sum_elem1 = 0
            sum_elem2 = 0 
            for l in range(n1rd, (n1rd + n2rd)):
                sum_elem0 += pix_val[l][0]
                sum_elem1 += pix_val[l][1]
                sum_elem2 += pix_val[l][2]
             
            img_array2.append([sum_elem0,sum_elem1,sum_elem2])            

            sum_elem0 = 0
            sum_elem1 = 0
            sum_elem2 = 0 
            for l in range((n1rd + n2rd), listlen):
                sum_elem0 += pix_val[l][0]
                sum_elem1 += pix_val[l][1]
                sum_elem2 += pix_val[l][2]
             
            img_array3.append([sum_elem0,sum_elem1,sum_elem2])

            temp_array_flat = []
            for k in range(0, len(img_array1)):
                lst1 = img_array1[k]
                for j in range(0, len(lst1)):
                    temp_array.append(lst1[j])
                                  
                lst2 = img_array2[k]
                for j in range(0, len(lst2)):
                    temp_array.append(lst2[j])
            
                lst3 = img_array3[k]
                for j in range(0, len(lst3)):
                    temp_array.append(lst3[j])
                    
            pxel_array.append(temp_array)
        except TypeError as err:
            print('Handling run-time error:', err)
            invalid_imgs.append(i) 

        #pxelArray = np.matrix(pxel_array)    
    
    return pxel_array, invalid_imgs

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def roundup(a, digits=0):
    #n = 10**-digits
    #return round(math.ceil(a / n) * n, digits)
    return round(a, digits)

#file = "C:/Users/CONALDES/Documents/WheatGrouthStage/Idx_test.csv"    
#if not os.path.isfile(file):
#    xyMatadata()

file = "C:/Users/CONALDES/Documents/WheatGrouthStage/Idx_test9.csv"    
if not os.path.isfile(file):
    xyMatadata()

#read csv file
#xy_train_df = pd.read_csv('C:/Users/CONALDES/Documents/WheatGrouthStage/xy_train.csv')
df = pd.read_csv('C:/Users/CONALDES/Documents/WheatGrouthStage/xy_train9.csv')
df = df.dropna()
#x_test_df = pd.read_csv('C:/Users/CONALDES/Documents/WheatGrouthStage/Idx_test.csv')
x_test_df = pd.read_csv('C:/Users/CONALDES/Documents/WheatGrouthStage/Idx_test9.csv')
x_test_df = x_test_df.dropna() #remove all lines with missing observations
test_x = x_test_df.drop('UID', axis=1)
test_UID = x_test_df['UID']
test_UID = test_UID.to_numpy()

print('test_UID', test_UID)
print('test_x', test_x)


# Reading Data 
#df = pd.read_csv('C:/Users/CONALDES/Documents/IrisData/Iris.csv')
#df.describe()
#df.info()

#df['GrowthStage']=df['GrowthStage']
#df['GrowthStage'] = df['GrowthStage'].map({'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6})
#df["GrowthStage"].unique()
df.head()

# Distances
def euclidian(p1, p2): 
    dist = 0
    for i in range(len(p1)):
        dist = dist + np.square(p1[i]-p2[i])
    dist = np.sqrt(dist)
    return dist

def manhattan(p1, p2): 
    dist = 0
    for i in range(len(p1)):
        dist = dist + abs(p1[i]-p2[i])
    return dist

def minkowski(p1, p2, q): 
    dist = 0
    for i in range(len(p1)):
        dist = dist + abs(p1[i]-p2[i])**q
    dist = np.sqrt(dist)**(1/q)
    return dist


# kNN Function
def kNN(X_train,y_train, X_test, k, dist='euclidian',q=2):
    pred = []
    # Adjusting the data type
    if isinstance(X_test, np.ndarray):
        X_test=pd.DataFrame(X_test)
    if isinstance(X_train, np.ndarray):
        X_train=pd.DataFrame(X_train)
        
    #print('kNN Function -> X_train', X_train)
    #print('kNN Function -> X_test', X_test)

    print('                                 ')
    print('len(X_test)', len(X_test))
    for i in range(len(X_test)):
        print('record -> ' + str(i) + ' of  X_test')
        # Calculating distances for our test point
        newdist = np.zeros(len(y_train))        # newdist.shape (10693,)
        
        #print('@@@@@@@@@@@@@@@ euclidian @@@@@@@@@@@@@@@')
        if dist=='euclidian':
            for j in range(len(y_train)):
                newdist[j] = euclidian(X_train.iloc[j,:], X_test.iloc[i,:])
                #print('j, newdist', j, newdist[j])
                      
        #print('@@@@@@@@@@@@@@@ manhattan @@@@@@@@@@@@@@@')
        if dist=='manhattan':
            for j in range(len(y_train)):
                newdist[j] = manhattan(X_train.iloc[j,:], X_test.iloc[i,:])
                #print('j, newdist', j, newdist[j])
                      
        #print('@@@@@@@@@@@@@@@ minkowski @@@@@@@@@@@@@@@')
        if dist=='minkowski':
            for j in range(len(y_train)):
                newdist[j] = minkowski(X_train.iloc[j,:], X_test.iloc[i,:],q)
                #print('j, newdist', j, newdist[j])

        # Merging actual labels with calculated distances
        newdist = np.array([newdist, y_train])

        ## Finding the closest k neighbors
        # Sorting index
        idx = np.argsort(newdist[0,:])

        # Sorting the all newdist
        newdist = newdist[:,idx]
        #print(newdist)

        #print('newdist', newdist)
        # We should count neighbor labels and take the label which has max count
        # Define a dictionary for the counts
        c = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0}
        # Update counts in the dictionary 
        for j in range(k):
            #print('j, newdist', j, newdist[1,j])
            #c[str(int(newdist[1,j]))] = c[str(int(newdist[1,j]))] + 1
            #print('j, newdist', j, newdist[0,j])
            c[str(int(newdist[0,j]))] = c[str(int(newdist[0,j]))] + 1

        key_max = max(c.keys(), key=(lambda k: c[k]))
        pred.append(int(key_max))
        
    return pred



# Sigmoid Function 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost Function
def J(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

# Gradient Descent Function
def gradientdescent(X, y, lmd, alpha, num_iter, print_cost):

    # select initial values zero
    theta = np.zeros(X.shape[1])
    
    costs = []  
    
    for i in range(num_iter):
        z = np.dot(X, theta)
        h = sigmoid(z)
        
        # adding regularization 
        reg = lmd / y.size * theta
        # first theta is intercept
        # it is not regularized
        reg[0] = 0
        cost = J(h, y)
        
        gradient = np.dot(X.T, (h - y)) / y.size + reg
        theta = theta - alpha * gradient
    
        if print_cost and i % 100 == 0: 
            print('Number of Iterations: ', i, 'Cost : ', cost, 'Theta: ', theta)
        if i % 100 == 0:
            costs.append(cost)
      
    return theta, costs

# Predict Function 
def predict(X_test, theta):
    z = np.dot(X_test, theta)
    return sigmoid(z)

# Main Logistic Function
def logistic(X_train, y_train, X_test, lmd=0, alpha=0.1, num_iter=30000, print_cost = False):
    # Adding intercept
    intercept = np.ones((X_train.shape[0], 1))
    X_train = np.concatenate((intercept, X_train), axis=1)
    
    intercept = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((intercept, X_test), axis=1)

    # one vs rest
    u=set(y_train)
    t=[]
    allCosts=[]   
    for c in u:
        # set the labels to 0 and 1
        ynew = np.array(y_train == c, dtype = int)
        theta_onevsrest, costs_onevsrest = gradientdescent(X_train, ynew, lmd, alpha, num_iter, print_cost)
        t.append(theta_onevsrest)
        
        # Save costs
        allCosts.append(costs_onevsrest)
        
    # Calculate probabilties
    pred_test = np.zeros((len(u),len(X_test)))
    for i in range(len(u)):
        pred_test[i,:] = predict(X_test,t[i])
    
    # Select max probability
    prediction_test = np.argmax(pred_test, axis=0)

    '''
    # Calculate probabilties
    pred_train = np.zeros((len(u),len(X_train)))
    for i in range(len(u)):
        pred_train[i,:] = predict(X_train,t[i])

    # Select max probability
    prediction_train = np.argmax(pred_train, axis=0)
    
    d = {"costs": allCosts,
         "Y_prediction_test": prediction_test, 
         "Y_prediction_train" : prediction_train, 
         "learning_rate" : alpha,
         "num_iterations": num_iter,
         "lambda": lmd}
    '''
    d = {"costs": allCosts,
         "Y_prediction_test": prediction_test,           
         "learning_rate" : alpha,
         "num_iterations": num_iter,
         "lambda": lmd}

    return d

#Logistic Regression from Neural Network Perspective

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Select initial values zero
def initialize_with_zeros(dim):
    return np.zeros((dim,1)), 0

def propagate(w, b, X, Y):
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T,X)+b) # compute activation
    cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)) # compute cost
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = 1/m*np.dot(X,(A-Y).T)
    db = 1/m*np.sum(A-Y)
    
    # keep grads in a dictionary 
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):    
    costs = []
    
    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        w = w-learning_rate*dw
        b = b-learning_rate*db 
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
            
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    # Save pameters and gradients
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


def predict_nn(w, b, X):    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities
    A = sigmoid(np.dot(w.T,X)+b)
        
    return A

#def model(X_train, Y_train, X_test, Y_test, num_iterations = 30000, learning_rate = 0.1, print_cost = False):
def model(X_train, Y_train, X_test, num_iterations = 30000, learning_rate = 0.1, print_cost = False):
    # pandas to numpy
            
    #X_train = X_train.values
    #Y_train = Y_train.values.reshape((1,Y_train.shape[0]))
    Y_train = Y_train.reshape((1,Y_train.shape[0]))
    #X_test = X_test.values
    
    #print('reshaped Y_train', Y_train)
    #print('reshaped Y_train', Y_train.shape)
    # take transpose of X
    X_train = X_train.T
    X_test = X_test.T
    
    # initialize parameters with zeros 
    w, b = initialize_with_zeros(X_train.shape[0])
    
    # one vs all
    u = set(y_train)
    param_w = []
    param_b = []
    allCosts = []
    for c in u:
        # set the labels to 0 and 1
        ynew = np.array(y_train == c, dtype = int)
        # Gradient descent 
        parameters, grads, costs = optimize(w, b, X_train, ynew, num_iterations, learning_rate, print_cost = print_cost)
        
        # Save costs
        allCosts.append(costs)
        
        # Retrieve parameters w and b from dictionary "parameters"
        param_w.append(parameters["w"])
        param_b.append(parameters["b"])
    
    # Calculate probabilties
    pred_test = np.zeros((len(u),X_test.shape[1]))
    for i in range(len(u)):
        pred_test[i,:] = predict_nn(param_w[i], param_b[i], X_test)
    
    # Select max probability
    Y_prediction_test = np.argmax(pred_test, axis=0)

    '''
    # Calculate probabilties
    pred_train = np.zeros((len(u),X_train.shape[1]))
    for i in range(len(u)):
        pred_train[i,:] = predict_nn(param_w[i], param_b[i], X_train)
    
    # Select max probability
    Y_prediction_train = np.argmax(pred_train, axis=0)
        
    d = {"costs": allCosts,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    '''
    d = {"costs": allCosts,
         "Y_prediction_test": Y_prediction_test, 
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


# I chose data points close to the real data points X[15], X[66] and X[130]
test = np.array([[3459509,3046864,1066346,3503969,3077538,1072323,3428571,3028468,1010139],[902362,1517847,114550,858426,1551314,137940,683828,1269139,142584]])
print("TEST POINTS\n", test)

all_X = df[['SumR1', 'SumG1', 'SumB1', 'SumR2', 'SumG2', 'SumB2', 'SumR3', 'SumG3', 'SumB3']]
all_y = df['GrowthStage']

# split data as training and test
#sepal_length	sepal_width	petal_length	petal_width	species
df=df[['SumR1', 'SumG1', 'SumB1', 'SumR2', 'SumG2', 'SumB2', 'SumR3', 'SumG3', 'SumB3','GrowthStage']]
train_data,test_data = train_test_split(df,train_size = 0.8,random_state=2)
print('train_data', train_data)
print('test_data', test_data)
X_train = train_data[['SumR1', 'SumG1', 'SumB1', 'SumR2', 'SumG2', 'SumB2', 'SumR3', 'SumG3', 'SumB3']]
y_train = train_data['GrowthStage']
X_test = test_data[['SumR1', 'SumG1', 'SumB1', 'SumR2', 'SumG2', 'SumB2', 'SumR3', 'SumG3', 'SumB3']]
y_test = test_data['GrowthStage']

X_train = X_train.to_numpy()
X_train = X_train.astype("float")

X_test = X_test.to_numpy()
X_test = X_test.astype("float")

y_train = y_train.to_numpy()
y_train = y_train.astype("int")

y_test = y_test.to_numpy()
y_test = y_test.astype("int")

test_x = test_x.to_numpy()
test_x = test_x.astype("float")

print('                         ')
print('X_train', X_train)
print('X_test', X_test)
print('y_train', y_train)
print('y_test', y_test)
print('                         ')

sc = StandardScaler()
X_train = pd.DataFrame(X_train)
X_train = sc.fit_transform(X_train)
X_test = pd.DataFrame(X_test)
X_test = sc.fit_transform(X_test)

test_x = pd.DataFrame(test_x)
test_x = sc.transform(test_x)

print('X_train', X_train)
print('X_test', X_test)
print('test_x', test_x)
print('                         ')
'''
def transform(i):
    if i == 0:
        return 'Iris-setosa'
    if i == 1:
        return 'Iris-versicolor'
    if i == 2:
        return 'Iris-virginica'
'''

def transform(i):
    if i == 0:
        return '1'
    if i == 1:
        return '2'
    if i == 2:
        return '3'
    if i == 3:
        return '4'
    if i == 4:
        return '5'
    if i == 5:
        return '6'
    if i == 6:
        return '7'

plt.figure(figsize=(10,10))
t=np.unique(all_y)

'''
ax1=plt.subplot(2, 2, 1)
ax1.set(xlabel='Sepal Length (cm)', ylabel='Sepal Width (cm)')
plt.plot(df[df['GrowthStage']==t[0]].iloc[:,0], df[df['GrowthStage']==t[0]].iloc[:,1], 'o', color='y')
plt.plot(df[df['GrowthStage']==t[1]].iloc[:,0], df[df['GrowthStage']==t[1]].iloc[:,1], 'o', color='r')
plt.plot(df[df['GrowthStage']==t[2]].iloc[:,0], df[df['GrowthStage']==t[2]].iloc[:,1], 'o', color='b')
# test datapoints
plt.plot(test[0,0],test[0,1],'*',color="k")
plt.plot(test[1,0],test[1,1],'*',color="k")
plt.plot(test[2,0],test[2,1],'*',color="k")

ax2=plt.subplot(2, 2, 2)
ax2.set(xlabel='Petal Length (cm)', ylabel='Petal Width (cm)')
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
plt.plot(df[df['GrowthStage']==t[0]].iloc[:,2], df[df['GrowthStage']==t[0]].iloc[:,3], 'o', color='y')
plt.plot(df[df['GrowthStage']==t[1]].iloc[:,2], df[df['GrowthStage']==t[1]].iloc[:,3], 'o', color='r')
plt.plot(df[df['GrowthStage']==t[2]].iloc[:,2], df[df['GrowthStage']==t[2]].iloc[:,3], 'o', color='b')
# test datapoints
plt.plot(test[0,2],test[0,3],'*',color="k")
plt.plot(test[1,2],test[1,3],'*',color="k")
plt.plot(test[2,2],test[2,3],'*',color="k")

ax3=plt.subplot(2, 2, 3)
ax3.set(xlabel='Sepal Length (cm)', ylabel='Petal Length (cm)')
plt.plot(df[df['GrowthStage']==t[0]].iloc[:,0], df[df['GrowthStage']==t[0]].iloc[:,2], 'o', color='y')
plt.plot(df[df['GrowthStage']==t[1]].iloc[:,0], df[df['GrowthStage']==t[1]].iloc[:,2], 'o', color='r')
plt.plot(df[df['GrowthStage']==t[2]].iloc[:,0], df[df['GrowthStage']==t[2]].iloc[:,2], 'o', color='b')
# test datapoints
plt.plot(test[0,0],test[0,2],'*',color="k")
plt.plot(test[1,0],test[1,2],'*',color="k")
plt.plot(test[2,0],test[2,2],'*',color="k")

ax4=plt.subplot(2, 2, 4)
ax4.set(xlabel='Sepal Width (cm)', ylabel='Petal Width (cm)')
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()
plt.plot(df[df['GrowthStage']==t[0]].iloc[:,1], df[df['GrowthStage']==t[0]].iloc[:,3], 'o', color='y')
plt.plot(df[df['GrowthStage']==t[1]].iloc[:,1], df[df['GrowthStage']==t[1]].iloc[:,3], 'o', color='r')
plt.plot(df[df['GrowthStage']==t[2]].iloc[:,1], df[df['GrowthStage']==t[2]].iloc[:,3], 'o', color='b')
# test datapoints
plt.plot(test[0,1],test[0,3],'*',color="k")
plt.plot(test[1,1],test[1,3],'*',color="k")
plt.plot(test[2,1],test[2,3],'*',color="k")
'''

'''
# Predicting the classes of the test data by kNN 
# Decide k value
k = 10
# print results
all_X = sc.fit_transform(all_X)
test = sc.transform(test)

print("k-NN ("+str(k)+"-nearest neighbors)\n")
c = kNN(all_X,all_y,test,k)
for i in range(len(c)):
    ct=set(map(transform,[c[i]]))
    print("Test point: "+str(test[i,:])+"  Label: "+str(c[i])+" "+str(ct))


#k-NN from Scratch vs scikit-learn k-NN

# k-NN from scratch
c=kNN(X_train,y_train,X_test,k)

try:
    cm=confusion_matrix(y_test, c)
            
    # logistic regression - scikit learn
    sck = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
    sck_cm=confusion_matrix(y_test, sck.predict(X_test))

    plt.figure(figsize=(15,6))
    plt.suptitle("Confusion Matrixes",fontsize=24)

    plt.subplot(1,2,1)
    plt.title("k-NN from Scratch")
    sns.heatmap(cm, annot = True, cmap="Greens",cbar=False);

    plt.subplot(1,2,2)
    plt.title("k-NN - scikit learn")
    sns.heatmap(sck_cm, annot = True, cmap="Greens",cbar=False)
except ValueError as val_err:
    print('Handling run-time error:', val_err)
'''

#Logistic Regression from Scratch

# Predicting the classes of the test data by Logistic Regression
print("Logistic Regression\n")
c=logistic(X_train,y_train,test)
# print results
for i in range(len(c['Y_prediction_test'])):
    ct=set(map(transform,[c['Y_prediction_test'][i]]))
    print("Test point: "+str(test[i,:])+"  Label: "+str(c['Y_prediction_test'][i])+" "+str(ct))


#Logistic Regression from Scratch vs Logistic Regression from Neural Network Perspective

# logistic regression from scratch
start=dt.datetime.now()
c=logistic(X_train,y_train,X_test)
# Print train/test Errors
#print('y_train', y_train)
#print('c["Y_prediction_train"]', c["Y_prediction_train"])
print('y_test', y_test)
print('c["Y_prediction_test"]', c["Y_prediction_test"])
#print('c["Y_prediction_train"] - y_train', c["Y_prediction_train"] - y_train)
print('c["Y_prediction_test"] - y_test', c["Y_prediction_test"] - y_test)

print('Elapsed time of logistic regression from scratch: ',str(dt.datetime.now()-start))
#print("train accuracy: {} %".format(100 - np.mean(np.abs(c["Y_prediction_train"] - y_train)) * 100))
print("test accuracy: {} %".format(np.abs(100 - np.mean(np.abs(c["Y_prediction_test"] - y_test)) * 100)))


# Logistic Regression from Neural Network Perspective
start=dt.datetime.now()
#d = model(X_train, y_train, X_test, y_test)
d = model(X_train, y_train, X_test)
#print('y_train', y_train)
#print('d["Y_prediction_train"]', d["Y_prediction_train"])
print('y_test', y_test)
print('d["Y_prediction_test"]', d["Y_prediction_test"])
#print('d["Y_prediction_train"] - y_train', d["Y_prediction_train"] - y_train)
print('d["Y_prediction_test"] - y_test', d["Y_prediction_test"] - y_test)
print('\nElapsed time of Logistic Regression from Neural Network Perspective: ',str(dt.datetime.now()-start))
#print("train accuracy: {} %".format(100 - np.mean(np.abs(d["Y_prediction_train"] - y_train)) * 100))
print("test accuracy: {} %".format(np.abs(100 - np.mean(np.abs(d["Y_prediction_test"] - y_test)) * 100)))

#try:
cm=confusion_matrix(y_test, c['Y_prediction_test'])

plt.figure(figsize=(15,6))
plt.suptitle("Confusion Matrixes",fontsize=24)

plt.subplot(1,2,1)
plt.title("Logistic Regression from Scratch")
sns.heatmap(cm, annot = True, cmap="Greens",cbar=False);
#except ValueError as val_err:
#    print('Handling run-time error:', val_err)

#try:
#cm=confusion_matrix(y_test, d['Y_prediction_test'].reshape(30,))
cm=confusion_matrix(y_test, d['Y_prediction_test'])

plt.subplot(1,2,2)
plt.title("Logistic Regression from Neural Network Perspective")
sns.heatmap(cm, annot = True, cmap="Greens",cbar=False)
#except ValueError as val_err:
#    print('Handling run-time error:', val_err)
    
# Learning rates
lr = [0.1, 0.01, 0.001]

logistic_preds = {}
for i in range(len(lr)):
    # Run the model for different learning rates
    c = logistic(X_train, y_train, test_x, alpha = lr[i])
    print("logistic -> Y_prediction_test, testln", c["Y_prediction_test"], len(c["Y_prediction_test"]))
    logistic_preds[i] = c["Y_prediction_test"]
    
    # Adjust results to plot    
    dfcost = pd.DataFrame(list(c['costs'])).transpose()
    #dfcost.columns = ['SumR1', 'SumG1', 'SumB1', 'SumR2', 'SumG2', 'SumB2', 'SumR3', 'SumG3', 'SumB3']
    
    # Plot the costs
    if i==0 : f, axes = plt.subplots(1, 3,figsize=(24,4))
    sns.lineplot(data = dfcost.iloc[:, :3], ax=axes[i])
    sns.despine(right=True, offset=True)
    axes[i].set(xlabel='Iterations (hundreds)', ylabel='Cost ' +'(Learning Rate: ' + str(lr[i]) + ')')
    
plt.suptitle("Logistic Regression from Scratch\n",fontsize=24);

model_preds = {}
for i in range(len(lr)):
    # Run the model for different learning rates
    #d = model(X_train, y_train, X_test, y_test, learning_rate = lr[i])
    d = model(X_train, y_train, test_x, learning_rate = lr[i])
    print("model -> Y_prediction_test, testln", d["Y_prediction_test"], len(d["Y_prediction_test"]))
    model_preds[i] =  d["Y_prediction_test"]
    
    # Adjust results to plot
    dfcost = pd.DataFrame(list(d['costs'])).transpose()
    #dfcost.columns = ['SumR1', 'SumG1', 'SumB1', 'SumR2', 'SumG2', 'SumB2', 'SumR3', 'SumG3', 'SumB3']
    
    # Plot the costs
    if i==0 : f, axes = plt.subplots(1, 3,figsize=(30,5))
    sns.lineplot(data = dfcost.iloc[:, :3], ax=axes[i])
    sns.despine(right=True, offset=True)
    axes[i].set(xlabel='Iterations (hundreds)', ylabel='Cost ' +'(Learning Rate: ' + str(lr[i]) + ')')
    
plt.suptitle("Logistic Regression from Neural Network Perspective\n",fontsize=24)
plt.show()

test_UID = np.vstack(test_UID) 

logistic_preds0 = np.vstack(logistic_preds[0])
logistic_preds1 = np.vstack(logistic_preds[1])
logistic_preds2 = np.vstack(logistic_preds[2])

model_preds0 = np.vstack(model_preds[0])
model_preds1 = np.vstack(model_preds[1])
model_preds2 = np.vstack(model_preds[2])

UIDs_logistic_preds0 = np.concatenate((test_UID, logistic_preds0), axis=1)
with open("C:/Users/CONALDES/Documents/WheatGrouthStage/Conaldes_logistic_preds0.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['UID', 'growth_stage']) 
    for row in UIDs_logistic_preds0:    
        l = list(row)    
        writer.writerow(l)

UIDs_logistic_preds1 = np.concatenate((test_UID, logistic_preds1), axis=1)
with open("C:/Users/CONALDES/Documents/WheatGrouthStage/Conaldes_logistic_preds1.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['UID', 'growth_stage']) 
    for row in UIDs_logistic_preds1:    
        l = list(row)    
        writer.writerow(l)

UIDs_logistic_preds2 = np.concatenate((test_UID, logistic_preds2), axis=1)
with open("C:/Users/CONALDES/Documents/WheatGrouthStage/Conaldes_logistic_preds2.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['UID', 'growth_stage']) 
    for row in UIDs_logistic_preds2:    
        l = list(row)    
        writer.writerow(l)

UIDs_model_preds0 = np.concatenate((test_UID, model_preds0), axis=1)
with open("C:/Users/CONALDES/Documents/WheatGrouthStage/Conaldes_model_preds0.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['UID', 'growth_stage']) 
    for row in UIDs_model_preds0:    
        l = list(row)    
        writer.writerow(l)

UIDs_model_preds1 = np.concatenate((test_UID, model_preds1), axis=1)
with open("C:/Users/CONALDES/Documents/WheatGrouthStage/Conaldes_model_preds1.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['UID', 'growth_stage']) 
    for row in UIDs_model_preds1:    
        l = list(row)    
        writer.writerow(l)

UIDs_model_preds2 = np.concatenate((test_UID, model_preds2), axis=1)
with open("C:/Users/CONALDES/Documents/WheatGrouthStage/Conaldes_model_preds2.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['UID', 'growth_stage']) 
    for row in UIDs_model_preds2:    
        l = list(row)    
        writer.writerow(l)
##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

