#Regression data Norconsult interview
regdata <- read.csv('Documents/Case/Norconsult/Maskinlæring Oppgave 02 Data/TrainAndValid.csv', header = TRUE, sep = ',')
set.seed(3) 
#Task 1: Simple statistics
mean(regdata$SalePrice)
max(regdata$SalePrice)
min(regdata$SalePrice)
median(regdata$SalePrice)
hist(regdata$SalePrice, breaks = 80, xlab = 'SalePrice', main = 'Histogram of SalePrice')
mean(regdata$ModelID)
max(regdata$ModelID)
min(regdata$ModelID)
median(regdata$ModelID)
hist(regdata$ModelID, breaks = 80, xlab = 'ModelID', main = 'Histogram of ModelID')

#Task 2: Missing values
sum(is.na(regdata))
regdata[!complete.cases(regdata),]
newdata = na.omit(regdata)
hist(newdata$SalePrice, breaks = 80)

#Task 3: Datatypes
regdata$saledate = sub(" 0:00", "", regdata$saledate)
regdata$saledate = as.Date(regdata$saledate, "%m/%d/%Y")
regdata = regdata[order(as.Date(regdata$saledate)), ]
plot(regdata$saledate, regdata$SalePrice, type = 'l')
newdata$saledate = sub(" 0:00", "", newdata$saledate)
newdata$saledate = as.Date(newdata$saledate, "%m/%d/%Y")
newdata = newdata[order(as.Date(newdata$saledate)), ]
plot(newdata$saledate, newdata$SalePrice, type = 'l')
plot(newdata$UsageBand, newdata$SalePrice)
plot(newdata$state, newdata$Saleprice)
smalldata = newdata[c(as.Date(newdata$saledate) > "2004-01-01") ,]

#Task 4: Model selection

smtrainsize = floor(0.8 * nrow(smalldata))
smtrain <- smalldata[1:smtrainsize, ]
smtest <- smalldata[-c(1:smtrainsize), ]
datamat = data.matrix(smalldata)
max = apply(datamat , 2 , max)
min = apply(datamat, 2 , min)
scaled = as.data.frame(scale(datamat, center = min, scale = max - min))
datamat = scaled
train_size <- floor(0.8 * nrow(datamat))
train <- datamat[c(1:train_size), ]
test <- datamat[-c(1:train_size), ]
#Look at correlations with SalePrice
corrmat = cor(datamat)
plot(1:53,abs(corrmat[2, ]), xlab = "Feature", ylab = "abs(Correlation)")
lines(1:53, rep(0.25,53))
corrind = integer()
for (j in 1:53){
  if (abs(corrmat[j ,2]) > 0.25){
    corrind = c(corrind,j)
  }
}
cordata = datamat[ ,corrind]
cortrain = train[ ,corrind]
cortest = test[ ,corrind]
#Create linear regression model
fullmodel = lm(SalePrice ~ ., as.data.frame(datamat))
fullpred = predict.lm(fullmodel, as.data.frame(test), se.fit = T)
cormodel = lm(SalePrice ~ ., as.data.frame(cortrain))
corpred = predict.lm(cormodel, as.data.frame(cortest), se.fit = TRUE)

#Other nonlinear methods
redtrain = 8000;
redtest = 2000;
library(randomForest)
rfmodel = randomForest(SalePrice ~ ., data = cortrain[1:redtrain, ], ntree = 1000, importance = TRUE)
rfpred = predict(rfmodel, cortest[1:redtest, ])
rfmse = mean((cortest[1:redtest ,1] - rfpred)^2)*(max[2]-min[2])

library(nnet)
library(neuralnet)
library(NeuralNetTools)
n <- names(cortrain)
f <- as.formula(paste("SalePrice ~", paste(n[!n %in% "SalePrice"], collapse = " + ")))
#Single layer
nnmod = nnet(SalePrice ~ ., data = as.data.frame(train), size = 15, maxit = 400, linout = T)
nnpred = predict(nnmod, test)
nnmse = mean((test[ ,2] - nnpred)^2)*(max[2]-min[2])

nnmod2 = nnet(f, data = as.data.frame(cortrain[1:redtrain, ]), size = 20, maxit = 500, linout = T)
nnpred2 = predict(nnmod2, cortest[1:redtest, ])
nnmse2 = mean((cortest[1:redtest ,1] - nnpred2)^2)*(max[2]-min[2])
plotnet(nnmod)
hist(nnpred, breaks = 50)

library(e1071)
svmmodel = svm(SalePrice ~ ., data=cortrain[1:redtrain, ], type = "eps", kernel = "radial")
svmpred = predict(svmmodel, cortest[1:redtest, ])
svmse = mean((cortest[1:redtest ,1] - svmpred)^2)*(max[2]-min[2])

#Task 5 & 6: Metric and Results
fullrmse = mean((test[ ,2] - fullpred$fit)^2)*(max[2]-min[2])
summary(fullmodel)

par(mfrow = c(2,1))
hist(test[ ,2], breaks = 80, xlim = c(0,0.8), xlab = 'True SalePrice', main = 'Testset SalePrice')
hist(fullpred$fit, breaks = 80, xlim = c(0,0.8), xlab = 'Predicted SalePrice', main = 'Predicted Saleprice')

cormse = sqrt(mean((cortest[ ,2] - corpred$fit)^2))*(max[2]-min[2])
summary(cormodel)

qqnorm(rstudent(fullmodel))
qqline(rstudent(fullmodel))
qqnorm(rstudent(cormodel))
qqline(rstudent(cormodel))

#Task 8: Overfitting, k-fold cross-validation
library(DAAG)
crossval = cv.lm(data=as.data.frame(train), form.lm = SalePrice ~ ., m=3)
