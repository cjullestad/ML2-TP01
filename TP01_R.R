rm(list=ls())
library(ISLR2)
data("Hitters")
attach(Hitters)

set.seed(1)
#A)Remove the observations for whom the salary information is unknown, and then log-transform the salaries.
Hitters <- na.omit(Hitters)
Hitters$Salary <- log(Hitters$Salary)

#B)Create a training set consisting of the first 200 observations, and a test set consisting of the remaining observations
train <- 1:200
test <- Hitters[-train,]
train <- Hitters[train,]

#C) Perform boosting on the training set with 1,000 trees for a range of values of the shrinkage parameter λ. 
# Produce a plot with different shrinkage values on the x-axis and the corresponding training set MSE on the y-axis

library(gbm)

pows = seq(-10, -0.2, by = 0.1)
lambdas = 10^pows #controls the rate at which it learns
train.err = rep(NA, length(lambdas))
for (i in 1:length(lambdas)) {
  boost.hitters = gbm(Salary ~ ., data = train, distribution = "gaussian", n.trees = 1000, shrinkage = lambdas[i])
  pred.train = predict(boost.hitters, train, n.trees = 1000)
  train.err[i] = mean((pred.train - train$Salary)^2)
}

plot(lambdas, train.err, type = "b", xlab = "Shrinkage values", ylab = "Training MSE")

#D)Produce a plot with different shrinkage values on the x-axis and the corresponding test set MSE on the y-axis.
test.err <- rep(NA, length(lambdas))
for (i in 1:length(lambdas)) {
  boost.hitters = gbm(Salary ~ ., data = train, distribution = "gaussian", n.trees = 1000, shrinkage = lambdas[i])
  yhat = predict(boost.hitters, test, n.trees = 1000)
  test.err[i] = mean((yhat - test$Salary)^2)
}
plot(lambdas, test.err, type = "b", xlab = "Shrinkage values", ylab = "Test MSE")

min(test.err) #boosting test mse

lambdas[which.min(test.err)] #shrinkage

#E)Compare the test MSE of boosting to the test MSE that results from applying two of the regression approaches seen in Chapters 3 and 6.
library(glmnet)
fit1 = lm(Salary ~ ., data = train)
pred1 = predict(fit1,test)
mean((pred1 - test$Salary)^2) #linear regression

x = model.matrix(Salary ~ ., data = train)
x.test = model.matrix(Salary ~ ., data = test)
y = train$Salary
fit2 = glmnet(x, y, alpha = 0)
pred2 = predict(fit2, newx = x.test)
mean((pred2 - test$Salary)^2)#lasso regression

#The test MSE for boosting is lower than for linear regression and ridge regression.

#F)Which variables appear to be the most important predictors in the boosted model?

boost.hitters <- gbm(Salary ~ ., data = train, distribution = "gaussian", n.trees = 1000, shrinkage = lambdas[which.min(test.err)])
summary(boost.hitters)
#“CAtBat” is by far the most important variable.

# G) Now apply bagging to the training set. What is the test set MSE for this approach?
library(randomForest)
bag.hitters <- randomForest(Salary ~ ., data = train, mtry = 19)
yhat.bag <- predict(bag.hitters, newdata = test)
mean((yhat.bag - test$Salary)^2) #bagging test mse

#The test MSE for bagging is 0.23, which is slightly lower than the test MSE for boosting.

