#Sigmod function 
sigmod = function(t){
  return(1 / (1 + exp(-t)))
}

#calculate the input t to the sigmod function for all samples. This is done via Matrix multiplication
#between the data matrix and the column vector containing the parameters. 
t_calc = function(x, lamda){
  no_lamda = length(lamda)
  zero_lamda = lamda[1]
  rest_lamda = lamda[2:no_lamda]
  x = data.matrix(x)
  all_t = zero_lamda + x %*% rest_lamda
  
  return(all_t)
}

#Function for the value of the objective function
objective = function(all_t, y){
  neg_lik = -log( y * sigmod(all_t) + (1 - y) * (1 - sigmod(all_t)) )
  return(sum(neg_lik))
}

#Function for the gradient of the objective function
gradient = function(all_t, x, y, lamda){
  cn = (y - (1 - y)) / (y * sigmod(all_t) + (1 - y) * (1 - sigmod(all_t)))
  grads_F = cn * sigmod(all_t) * (1 - sigmod(all_t))
  grad_l0 = -sum(grads_F)
  grad_l1n = numeric(length(lamda) - 1)
  
  for (i in seq(1, length(grad_l1n))) {
    grad_l1n[i] = -sum(grads_F * x[, i])
  }
  
  grad_l0n = c(grad_l0, grad_l1n)
  return(grad_l0n)
}

#Function to asses model accuracy.
score = function(x, y, lamda){
  T = t_calc(x, lamda)
  prob = sigmod(T)
  yhat = numeric(nrow(x))
  for (i in 1:nrow(x)) {
    if (prob[i] < 0.5){
      yhat[i] = 0
    }else{
      yhat[i] = 1}
  }
  error_rate = sum((y - yhat)^2)/nrow(x)
  return(error_rate)
}



###--------------------------------------------------------------------------###
### DATA PREPROCESSING 
# Now load in the auto data set.
auto = read.table('car.txt', header = T,na.strings = '?')

# Create label high.
attach(auto)
High = ifelse(mpg <= 23, 0, 1)

# Create dummy variables for the categorical origin variable 
origin1 = ifelse(origin != 1, 0, 1)
origin2 = ifelse(origin != 2, 0, 1)
origin3 = ifelse(origin != 3, 0, 1)
# Add high and dummy variables to auto
auto = data.frame(auto, origin1, origin2, origin3, High)

#Get the variables and label used to train the model
auto_logist = auto[c('horsepower', 'weight', 'year', 'origin1', 'origin2', 'origin3', 'High')]
#remove all rows which contain Na's
auto_logist = auto_logist[complete.cases(auto_logist),]
#Standardize all non-normal features   
auto_logist = data.frame(scale(auto_logist[,1:3]), auto_logist[c('origin1', 'origin2', 'origin3', 'High')])
#Get Train and test set
train = sample(1:nrow(auto_logist), nrow(auto_logist)/2)
x_train = auto_logist[train, 1:6]
y_train = auto_logist[train, 7]
x_test = auto_logist[-train, 1:6]
y_test = auto_logist[-train, 7]



### -------------------------------------------------------------------------###
### Calculate train and test scores for a range of iterations and learning rate. 
N = c(10, 100, 1000, 10000)
lr = c(10^-1, 10^-2, 10^-3, 10^-4, 10^-5, 10^-6, 10^-7, 10^-8, 10^-9, 10^-10)
train_scores = matrix(0, 10, 4)
test_scores = matrix(0, 10, 4)

for (z in 1:length(N)) {
  for (j in 1:length(lr)){
    lam = runif(7, -0.7, 0.7) # Calculate random lambda in the range [-0.7, 0.7]
    T = t_calc(x_train, lam)  # Calculate initial Sigmod input 
    #min_lik = numeric(N[z])
    
    for (i in 1:N[z]) {
      #min_lik[i] = objective(T, y_train)
      grad = gradient(T, x_train, y_train, lam) #Calculate gradient 
      lam = lam - lr[j] * grad #Calculate new lambda given gradient 
      T = t_calc(x_train, lam) #Calculate new sigmod input given lambda 
    }
    # Store train and test score given lambda
    train_scores[j, z] = score(x_train, y_train, lam)
    test_scores[j, z] = score(x_test, y_test, lam)
  }
}
# Format the train and test score matrices 
row.names(train_scores)= c('10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6',
                           '10^-7', '10^-8', '10^-9', '10^-10')
colnames(train_scores) = c('10', '100', '1000', '10000')
row.names(test_scores)= c('10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6',
                           '10^-7', '10^-8', '10^-9', '10^-10')
colnames(test_scores) = c('10', '100', '1000', '10000')

#Print train and test matrices  
print('The train score matrix for fixed stoping rule is:')
print(train_scores)
print('The test score matrix for fixed stoping rule is:')
print(test_scores)

#Now see how learning rate changes for given N 
#Put all 4 plots on one image to allow for easy comparison 
par(mfrow=c(2,2))

#Plot for N = 10
plot(1:10, train_scores[, 1], type = 'l', xlab = 'Lr = 10^(-x)', ylab = 'Error rate', main = 'N = 10',
     col = 'red')
lines(1:10, test_scores[, 1], col = 'blue', lty = 2)

#Plot for N = 100
plot(1:10, train_scores[, 2], type = 'l', xlab = 'Lr = 10^(-x)', ylab = 'Error rate', main = 'N = 100',
     col = 'red')
lines(1:10, test_scores[, 2], col = 'blue', lty = 2)

#Plot for N=1000
plot(1:10, train_scores[, 3], type = 'l', xlab = 'Lr = 10^(-x)', ylab = 'Error rate', main = 'N = 1000',
     col = 'red')
lines(1:10, test_scores[, 3], col = 'blue', lty = 2)

#Plot for N = 10000
plot(1:10, train_scores[, 4], type = 'l', xlab = 'Lr = 10^(-x)', ylab = 'Error rate', main = 'N = 10000',
     col = 'red')
lines(1:10, test_scores[, 4], col = 'blue', lty = 2)

#See how error rate changes with lr = 1/10000.
par(mfrow = c(1,1))
plot(1:4, train_scores[3, ], type = 'l', xlab = '10^x', ylab = 'Error rate', main = 'Lr = 1/10000',
     col = 'Red')
lines(1:4, test_scores[3, ], col = 'blue', lty = 2)



###--------------------------------------------------------------------------###
###Now try a different stopping rule 
# Calculate train and test scores for a range of iterations and learning rate. 
N = c(0.1, 0.01, 0.001, 0.0001)
lr = c(10^-1, 10^-2, 10^-3, 10^-4, 10^-5, 10^-6, 10^-7, 10^-8, 10^-9, 10^-10)
train_scores = matrix(0, length(lr), length(N))
test_scores = matrix(0, length(lr), length(N))
No_iter = matrix(0, length(lr), length(N))

for (z in 1:length(N)) {
  for (j in 1:length(lr)){
    # Calculate random lamda in the range [-0.7, 0.7]
    lam = runif(7, -0.7, 0.7)
    T = t_calc(x_train, lam)
    min_lik = numeric(10000)
    
    for (i in 1:10000) {
      grad = gradient(T, x_train, y_train, lam)
      lam = lam - lr[j] * grad
      T = t_calc(x_train, lam)
      min_lik[i] = objective(T, y_train)
      # keep count of current index
      iter = i 
      if (iter > 10){
        # change in minimum likelihood over 10 steps 
        min_lik_change = abs((min_lik[i] / min_lik[i - 10]) - 1)
        if (min_lik_change < N[z]){
          break
        }
      }
    }
    No_iter[j, z] = iter 
    train_scores[j, z] = score(x_train, y_train, lam)
    test_scores[j, z] = score(x_test, y_test, lam)
  }
}

row.names(No_iter)= c('10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6',
                      '10^-7', '10^-8', '10^-9', '10^-10')
colnames(No_iter) = c('0.1', '0.01', '0.001', '0.0001')
row.names(train_scores)= c('10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6',
                           '10^-7', '10^-8', '10^-9', '10^-10')
colnames(train_scores) = c('0.1', '0.01', '0.001', '0.0001')
row.names(test_scores)= c('10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6',
                          '10^-7', '10^-8', '10^-9', '10^-10')
colnames(test_scores) = c('0.1', '0.01', '0.001', '0.0001')

print('The number of iterations to each stoping rule is:')
print(No_iter)
print('The train score matrix for stoping rule based on the objective function is:')
print(train_scores)
print('The test score matrix based on the stopping rule')
print(test_scores)



###--------------------------------------------------------------------------###
#100 realizations for a learning rate of 0.01 and a stopping rule based on a 
#change in the objective function of 0.001.
lr = 0.01
N = 0.001
test_scores = numeric(length = 100)

for (j in 1:100){
  #Get a new train and test set
  #Get Train and test set
  train = sample(1:nrow(auto_logist), nrow(auto_logist)/2)
  x_train = auto_logist[train, 1:6]
  y_train = auto_logist[train, 7]
  x_test = auto_logist[-train, 1:6]
  y_test = auto_logist[-train, 7]
  
  # Calculate random lamda in the range [-0.7, 0.7]
  lam = runif(7, -0.7, 0.7)
  T = t_calc(x_train, lam)
  min_lik = numeric(10000)
  
  for (i in 1:10000) {
    grad = gradient(T, x_train, y_train, lam)
    lam = lam - lr * grad
    T = t_calc(x_train, lam)
    min_lik[i] = objective(T, y_train)
    # keep count of current index
    iter = i 
    if (iter > 10){
      # change in minimum likelihood over 10 steps 
      min_lik_change = abs((min_lik[i] / min_lik[i - 10]) - 1)
      if (min_lik_change < N){
        break
      }
    }
  }
  test_scores[j] = score(x_test, y_test, lam)
}
boxplot(test_scores, ylab = 'Error rate')


#Average of 4 models run 100 times. 
lr = 0.01
N = 0.001
test_scores = numeric(length = 100)

for (z in 1:4) {
  for (j in 1:100){
    #Get a new train and test set
    #Get Train and test set
    train = sample(1:nrow(auto_logist), nrow(auto_logist)/2)
    x_train = auto_logist[train, 1:6]
    y_train = auto_logist[train, 7]
    x_test = auto_logist[-train, 1:6]
    y_test = auto_logist[-train, 7]
    
    # Calculate random lambda in the range [-0.7, 0.7]
    lam = runif(7, -0.7, 0.7)
    T = t_calc(x_train, lam)
    min_lik = numeric(10000)
    
    for (i in 1:10000) {
      grad = gradient(T, x_train, y_train, lam)
      lam = lam - lr * grad
      T = t_calc(x_train, lam)
      min_lik[i] = objective(T, y_train)
      # keep count of current index
      iter = i 
      if (iter > 10){
        # change in minimum likelihood over 10 steps 
        min_lik_change = abs((min_lik[i] / min_lik[i - 10]) - 1)
        if (min_lik_change < N){
          break
        }
      }
    }
    #Find the sum of 4 the error rate of 4 different models
    test_scores[j] = test_scores[j] + score(x_test, y_test, lam)
  }
}
# Find the average of the error rate of 4 different models 
test_scores = test_scores / 4

boxplot(test_scores, ylab = 'Error rate')

###--------------------------------------------------------------------------###
###FINDING THE OPTIMAL MODEL 

#Create lists to store the 5 train and test score matrices for each stopping rule type
all_train_scores = list()
all_test_scores = list()
all_train_scores2 = list()
all_test_scores2 = list()
for(k in 1:5){
  #get a fresh test and train sample
  train = sample(1:nrow(auto_logist), nrow(auto_logist)/2)
  x_train = auto_logist[train, 1:6]
  y_train = auto_logist[train, 7]
  x_test = auto_logist[-train, 1:6]
  y_test = auto_logist[-train, 7]
  
  
  #get train and test scores for a fixed satoping rule
  N = c(10, 100, 1000, 10000)
  lr = c(10^-1, 10^-2, 10^-3, 10^-4, 10^-5, 10^-6, 10^-7, 10^-8, 10^-9, 10^-10)
  train_scores = matrix(0, 10, 4)
  test_scores = matrix(0, 10, 4)
  
  for (z in 1:length(N)) {
    for (j in 1:length(lr)){
      lam = runif(7, -0.7, 0.7) # Calculate random lambda in the range [-0.7, 0.7]
      T = t_calc(x_train, lam)  # Calculate initial Sigmod input 
      #min_lik = numeric(N[z])
      
      for (i in 1:N[z]) {
        #min_lik[i] = objective(T, y_train)
        grad = gradient(T, x_train, y_train, lam) #Calculate gradient 
        lam = lam - lr[j] * grad #Calculate new lambda given gradient 
        T = t_calc(x_train, lam) #Calculate new sigmod input given lambda 
      }
      # Store train and test score given lambda
      train_scores[j, z] = score(x_train, y_train, lam)
      test_scores[j, z] = score(x_test, y_test, lam)
    }
  }
  #Store in relevant list
  all_train_scores[[k]] = train_scores
  all_test_scores[[k]] = test_scores
  
  #Now find train and test scores with learning rule based on objective function
  N = c(0.1, 0.01, 0.001, 0.0001)
  lr = c(10^-1, 10^-2, 10^-3, 10^-4, 10^-5, 10^-6, 10^-7, 10^-8, 10^-9, 10^-10)
  train_scores2 = matrix(0, length(lr), length(N))
  test_scores2 = matrix(0, length(lr), length(N))
  No_iter = matrix(0, length(lr), length(N))
  
  for (z in 1:length(N)) {
    for (j in 1:length(lr)){
      # Calculate random lamda in the range [-0.7, 0.7]
      lam = runif(7, -0.7, 0.7)
      T = t_calc(x_train, lam)
      min_lik = numeric(10000)
      
      for (i in 1:10000) {
        grad = gradient(T, x_train, y_train, lam)
        lam = lam - lr[j] * grad
        T = t_calc(x_train, lam)
        min_lik[i] = objective(T, y_train)
        # keep count of current index
        iter = i 
        if (iter > 10){
          # change in minimum likelihood over 10 steps 
          min_lik_change = abs((min_lik[i] / min_lik[i - 10]) - 1)
          if (min_lik_change < N[z]){
            break
          }
        }
      }
      train_scores2[j, z] = score(x_train, y_train, lam)
      test_scores2[j, z] = score(x_test, y_test, lam)
    }
    all_train_scores2[[k]] = train_scores2
    all_test_scores2[[k]] = test_scores2
  }
  
  
}
#Get the average train and test scores
train_scores = (all_train_scores[[1]] + all_train_scores[[2]] + all_train_scores[[3]] 
                + all_train_scores[[4]] + all_train_scores[[5]]) / 5
test_scores = (all_test_scores[[1]] + all_test_scores[[2]] + all_test_scores[[3]]
               + all_test_scores[[4]] + all_test_scores[[5]]) / 5
train_scores2 = (all_train_scores2[[1]] + all_train_scores2[[2]] + all_train_scores2[[3]] 
                 + all_train_scores2[[4]] + all_train_scores2[[5]]) / 5
test_scores2 = (all_test_scores2[[1]] + all_test_scores2[[2]] + all_test_scores2[[3]]
                + all_test_scores2[[4]] + all_test_scores2[[5]]) / 5

# Format the train and test score matrices 
row.names(train_scores)= c('10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6',
                           '10^-7', '10^-8', '10^-9', '10^-10')
colnames(train_scores) = c('10', '100', '1000', '10000')
row.names(test_scores)= c('10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6',
                          '10^-7', '10^-8', '10^-9', '10^-10')
colnames(test_scores) = c('10', '100', '1000', '10000')
row.names(train_scores2)= c('10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6',
                            '10^-7', '10^-8', '10^-9', '10^-10')
colnames(train_scores2) = c('0.1', '0.01', '0.001', '0.0001')
row.names(test_scores2)= c('10^-1', '10^-2', '10^-3', '10^-4', '10^-5', '10^-6',
                           '10^-7', '10^-8', '10^-9', '10^-10')
colnames(test_scores2) = c('0.1', '0.01', '0.001', '0.0001')

#Print out train and test score matrices 
print('The train and test scores based on a fixed stoping rule are:')
print(train_scores)
print(test_scores)
print('The train and test scores with stoping rule based on the objective function are:')
print(train_scores2)
print(test_scores2)

## Find optimal lambda for N = 0.001 and lr = 0.001
lambda = matrix(0, 25, 7) #matrix to store all lambda 

for(j in 1:25){
  #get a fresh test and train sample
  train = sample(1:nrow(auto_logist), nrow(auto_logist)/2)
  x_train = auto_logist[train, 1:6]
  y_train = auto_logist[train, 7]
  x_test = auto_logist[-train, 1:6]
  y_test = auto_logist[-train, 7]
  
  # Calculate random lambda in the range [-0.7, 0.7]
  lam = runif(7, -0.7, 0.7)
  T = t_calc(x_train, lam)
  min_lik = numeric(10000)
  
  for (i in 1:10000) {
    grad = gradient(T, x_train, y_train, lam)
    lam = lam - 0.001 * grad
    T = t_calc(x_train, lam)
    min_lik[i] = objective(T, y_train)
    # keep count of current index
    iter = i 
    if (iter > 10){
      # change in minimum likelihood over 10 steps 
      min_lik_change = abs((min_lik[i] / min_lik[i - 10]) - 1)
      if (min_lik_change < 0.001){
        break
      }
    }
  }
  lambda[j, ] = lam
}

#Find the optimal lambda by finding the mean of the cols of the lambda matrix 
optimal_lambda = numeric(7)

for (i in 1:7){
  optimal_lambda[i] = mean(lambda[, i])
}

print('The optimal lambda is:')
print(optimal_lambda)