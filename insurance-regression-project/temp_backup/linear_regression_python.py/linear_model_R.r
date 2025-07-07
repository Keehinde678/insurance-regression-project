install.packages("tidyverse")



library(tidyverse)



install.packages(c("ggplot2", "dplyr", "readr"))
library(ggplot2)
library(dplyr)
library(readr)


install.packages("corrplot")



library(corrplot)



# Load necessary packages
library(tidyverse)     # for data manipulation and visualization
library(ggplot2)       # for plotting
library(corrplot)      # for correlation plot
library(caret)         # for model evaluation
library(psych)         # for descriptive stats


install.packages("caret")
library(caret)


install.packages("psych")
library(psych)


# Load necessary packages
library(tidyverse)     # for data manipulation and visualization
library(ggplot2)       # for plotting
library(corrplot)      # for correlation plot
library(caret)         # for model evaluation
library(psych)         # for descriptive stats


# Correct file path format — use either double backslashes or forward slashes
insurance <- read.csv("C:\\Users\\kehin\\Downloads\\archive (3)\\insurance.csv")
# Or:
# insurance <- read.csv("C:/Users/kehin/Downloads/archive (3)/insurance.csv")

# Now preview the data correctly by passing the data frame, not a string
head(insurance)
str(insurance)
summary(insurance)



# Check for missing values
colSums(is.na(insurance_data))


describe(insurance_data)


# Histogram for charges
ggplot(insurance_data, aes(x = charges)) +
  geom_histogram(fill = "lightblue", bins = 30) +
  theme_minimal() +
  labs(title = "Distribution of Insurance Charges", x = "Charges", y = "Count")


# Charges by smoking status
ggplot(insurance_data, aes(x = smoker, y = charges, fill = smoker)) +
  geom_boxplot() +
  labs(title = "Charges by Smoking Status") +
  theme_minimal()


# Charges by region
ggplot(insurance_data, aes(x = region, y = charges, fill = region)) +
  geom_boxplot() +
  labs(title = "Charges by Region") +
  theme_minimal()


# Encode categorical variables
insurance_data <- insurance_data %>%
  mutate(
    sex = factor(sex),
    smoker = factor(smoker),
    region = factor(region)
  )


# Numeric correlation
numeric_vars <- insurance_data %>%
  select_if(is.numeric)
corr_matrix <- cor(numeric_vars)

corrplot(corr_matrix, method = "number", type = "upper", tl.cex = 0.8)


# Fit the full linear regression model
model <- lm(charges ~ age + sex + bmi + children + smoker + region, data = insurance_data)
summary(model)


# Diagnostic plots
par(mfrow = c(2, 2))
plot(model)


insurance_data$predicted_charges <- predict(model, insurance_data)

# Plot actual vs predicted
ggplot(insurance_data, aes(x = charges, y = predicted_charges)) +
  geom_point(alpha = 0.5, color = "darkgreen") +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(title = "Actual vs Predicted Charges", x = "Actual", y = "Predicted") +
  theme_minimal()


# Model with interaction
model2 <- lm(charges ~ age + bmi*smoker + children + sex + region, data = insurance_data)
summary(model2)


# Plot 1
plot_charges <- ggplot(insurance, aes(x = charges)) +
  geom_histogram(fill = "lightblue", bins = 30) +
  theme_minimal() +
  labs(title = "Distribution of Insurance Charges")

ggsave("plot_charges.png", plot = plot_charges, width = 6, height = 4, dpi = 300)

# Plot 2
plot_smoker <- ggplot(insurance, aes(x = smoker, y = charges, fill = smoker)) +
  geom_boxplot() +
  labs(title = "Charges by Smoking Status") +
  theme_minimal()

ggsave("plot_smoker.png", plot = plot_smoker, width = 6, height = 4, dpi = 300)



