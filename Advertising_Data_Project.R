data=read.csv("C:/Users/prach/Downloads/Advertising_Data.csv")
data

anova(model)
# Perform exploratory data analysis (EDA)
summary(data)
head(data)
str(data)

# Fit a linear regression model
model <- lm( Product_Sold ~TV+ Billboards+Google_Ads+ Social_Media+Influencer_Marketing+ Affiliate_Marketing, data = data)
summary(model)

# Check assumptions of linear regression
# Extract residuals from the model
residuals <- residuals(model)

# Visual inspection
# Histogram
hist(residuals, main = "Histogram of Residuals")

# Q-Q plot
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals)

# Statistical test
shapiro.test(residuals)




# Assessing the impact of different marketing channels on product sales
cor(data[, c("TV", "Billboards", "Google_Ads", "Social_Media", "Influencer_Marketing", "Affiliate_Marketing")], data$Product_Sold)


#predictions <- predict(model,data = new_data)

# View the predictions
#predictions

tv=(data$TV)
tv
subset(-1)
product_sold=data$Product_Sold
product_sold
t1=t.test(tv,product_sold)
billboards=data$Billboards
billboards
t2=t.test(billboards,product_sold)
google_ads=data$Google_Ads
t3=t.test(google_ads,product_sold)
t4=t.test(data$Social_Media,product_sold)
t5=t.test(data$Influencer_Marketing,product_sold)
t6=t.test(data$Affiliate_Marketing,product_sold)



rs=summary(model)$r.squared


d1=data$TV
d2=data$Product_Sold
model=lm(Product_Sold ~TV,data=data)
s=summary(model)
rs=summary(model)$r.squared


d2=data$Billboards
model=lm(Product_Sold ~Billboards,data=data)
s=summary(model1)
rs=s$r.squared

d3=lm(Product_Sold ~Google_Ads,data=data)
s=summary(m3)
rs=s$r.squared

m1=lm(Product_Sold~Social_Media,data=data)
summary(m1)
r1=summary(m1)$r.squared
m2=lm(Product_Sold~Influencer_Marketing,data=data)
summary(m2)
r2=summary(m2)$r.squared
m3=lm(Product_Sold~data$Affiliate_Marketing,data=data)
summary(m3)
r3=summary(m3)$r.squared
#subset selection method

library(stats)
library(leaps)
regfit.full=regsubsets(Product_Sold~.,data=data)
summary(regfit.full)








#multicolaniarity
##v=vif(model)

#vif=(1/(1-rs))

library(olsrr)
o=ols_vif_tol(model)
library(car)

v=vif(model)
b=barplot(v,main="VIF values",horiz=TRUE,col="red")

reg=data.frame(data)
reg
#finding correlation plot
corre=cor(reg)
corre
library(corrplot)
corrplot(corre,method="circle",bg="grey")


# Extract residuals and fitted values from the model
residuals <- residuals(model)
fitted_values <- fitted(model)

# Plot residuals vs. fitted values
plot(fitted_values, residuals,
     xlab = "Fitted Values", ylab = "Residuals",
     main = "Residuals vs. Fitted Values Plot")
abline(h = 0, col = "red", lty = 2)  # Add a horizontal reference line at 0




# Extract independent variables from the model
independent_variables <- model$model[, -1]  # Exclude the intercept column

# Plot residuals vs. each independent variable
par(mfrow = c(2, 2))  # Setting up a 2x2 plot layout
for (i in 1:ncol(independent_variables)) {
  plot(independent_variables[, i], residuals,
       xlab = colnames(independent_variables)[i], ylab = "Residuals",
       main = paste("Residuals vs.", colnames(independent_variables)[i]))
  abline(h = 0, col = "red", lty = 2)  # Add a horizontal reference line at 0
}


library(corrplot)
corrplot(cor(model))















# Load necessary libraries
library(readr)
library(caret)
library(ggplot2)

# Load the dataset
#data <- read_csv("your_dataset.csv")

# Splitting the dataset into features (X) and target variable (y)
X <- subset(data, select = -Product_Sold)  # Features
y <- data$Product_Sold  # Target variable

# Splitting the dataset into training and testing sets
set.seed(42)  # for reproducibility
trainIndex <- createDataPartition(y, p = .8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]
length(y)
# Training the model
model <-lm(y ~x,cbind(y_train, X_train))

# Predicting the values
y_pred <- predict(model, newdata = X_test)

# Visualizing the predictions
ggplot() +
  geom_point(aes(x = y_test, y = y_pred), color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(x = "Actual Product Sold", y = "Predicted Product Sold") +
  ggtitle("Actual vs Predicted Product Sold")








y=(data$Product_Sold)
model2=lm(y~.,data=data)
library(olsrr)
ols_step_all_possible(model2)
k=ols_step_best_subset(model2)
k$metrics
p=plot(k)





# Load required libraries
library(caret)
library(randomForest)

# Load the provided dataset
#data <- read.csv("product_ad_data.csv")

# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
train_index <- createDataPartition(data$Product_Sold, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Train a random forest regression model
model <- train(Product_Sold ~ ., data = train_data, method = "rf")
# Train the KNN model using the training data
# Assuming your target variable is "Product_Sold" and other columns are features
library(class)
k <- 5 # choose the value of k
predicted <-knn(train = train_data[, -ncol(train_data)], test = test_data[, -ncol(test_data)], cl = train_data[, ncol(train_data)], k = k)

# Evaluate the model using the testing data
accuracy <- sum(predicted == test_data[, ncol(test_data)]) / nrow(test_data)
print(paste("Accuracy:", accuracy))
library(glmnet)
# Train a logistic regression model using the training data
model <- glm(Product_Sold ~ ., data = train_data, family = binomial)

# Make predictions using the logistic regression model
predicted_logreg <- predict(model, newdata = test_data, type = "response") > 0.5

# Calculate accuracy of logistic regression model
accuracy_logreg <- sum(predicted_logreg == test_data[, "Product_Sold"]) / nrow(test_data)
print(paste("Accuracy of Logistic Regression model:", accuracy_logreg))

# Compare accuracies of both models
if (accuracy_knn > accuracy_logreg) {
  print("KNN model has higher accuracy.")
} else if (accuracy_knn < accuracy_logreg) {
  print("Logistic Regression model has higher accuracy.")
} else {
  print("Both models have the same accuracy.")
}

# Function to predict regressors for given input data
predict_regressors <- function(input_data) {
  predicted_regressors <- predict(model, newdata = 500,600)
  return(predicted_regressors)
}

# Example usage:
new_data <- data.frame(
  TV = c(500, 600),
  Billboards = c(200, 300),
  Google_Ads = c(400, 500),
  Social_Media = c(300, 400),
  Influencer_Marketing = c(100, 200),
  Affiliate_Marketing = c(200, 300)
)
predicted_amounts <- predict_regressors(new_data)
print(predicted_amounts)





# Load required libraries
library(caret) # for data preprocessing and model evaluation
library(ggplot2) # for visualization

# Load your dataset
#data <- read.csv()

# Preprocess your data (e.g., handle missing values, encode categorical variables, normalize data)
# For example, if you have missing values:
# data <- na.omit(data)

# Split your dataset into training and testing sets (e.g., 70% training, 30% testing)
set.seed(123) # for reproducibility
train_index <- createDataPartition(data$Product_Sold, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Train the KNN model using the training data
k <- 5 # choose the value of k
knn_model <- train(Product_Sold ~ ., data = train_data, method = "knn", trControl = trainControl(method = "cv", number = 10), tuneGrid = expand.grid(k = k))
accuracy <- mean(knn_model == test_data$Product_Sold)
# Train the multiple linear regression model using the training data
lm_model <- train(Product_Sold ~ ., data = train_data, method = "lm", trControl = trainControl(method = "cv", number = 10))

# Compare model performances using resampling
compare_models <- resamples(list(KNN = knn_model, MultipleRegression = lm_model))

# Summarize the results
summary(compare_models)

# Visualize the results
dotplot(compare_models)














# Rename columns in new_data1 to match the original dataset
n_new_data1<- c("TV", "Billboards", "Google_Ads", "Social_Media", "Influencer_Marketing", "Affiliate_Marketing")
# Convert new_data1 to a data frame if it's not already
new_data1 <- data.frame(n_new_data1)

# Now try to predict using the model
predictions <- predict(model, newdata = new_data1)

# Now try to predict using the model
predictions <- predict(model, newdata =n_new_data1)

new_data1=data.frame(product_sold=c(890,350,650,230,500,150))
predict(model,newdata=new_data1)
predict(model,newdata=new_data1,interval="confidence")
predict(model,newdata=new_data1,interval="prediction",level=0.95)
pred_int=predict(model,interval="prediction",level=0.95)








# Assuming you have already loaded your dataset into a variable called 'data'

# Split the data into training and test sets (e.g., using the 'caret' package)
library(caret)
set.seed(123) # for reproducibility
train_index <- createDataPartition(data$target_variable, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Train the k-NN model
library(class) # for k-NN algorithm
k <- 5 # number of neighbors
knn_model <- knn(train = train_data[, predictors], test = test_data[, predictors], cl = train_data$target_variable, k = k)

# Evaluate accuracy
accuracy <- mean(knn_model == test_data$target_variable)
print(paste("Accuracy of k-NN model:", round(accuracy * 100, 2), "%"))




length(knn_model)
length(test_data$Product_Sold)
# Subset or reshape knn_model to match the length of test_data$Product_Sold
knn_model_subset <- knn_model[1:length(test_data$Product_Sold)]