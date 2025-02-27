# -Multiple-Linear-Regression
The first step involves importation of the different libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_regression import LinearRegression

# Define the independent variables (House Size, Bedrooms, House Age)
house_features = np.array([
    [50, 2, 10],  
    [75, 3, 5],  
    [100, 3, 8],  
    [125, 4, 4],  
    [150, 4, 2]
])
# Define the dependent variable (House Price in $)
house_prices = np.array([150000, 200000, 250000, 300000, 350000])

The third step in this case involves training the multiple linear regression
#initialize the linear regression model 
model = LinearRegression()
#After initializing train the model using the data set defined earlier. 
model.fit(house_features, house prices)

# Get coefficients and intercept
m_values = model.coef_  # Slopes
b = model.intercept_  # Intercept
# Display the equation
print(f"Equation of the regression line: Y = {m_values[0]:.2f}X1 + {m_values[1]:.2f}X2 + {m_values[2]:.2f}X3 + {b:.2f}")


# New house data for prediction (Size, Bedrooms, Age)
new_house = np.array([[120, 3, 6]])
# Predict the price
predicted_price = model.predict(new_house)
# Display prediction
print(f"Predicted House Price: ${predicted_price[0]:.2f}")

It is important to visualize the relationship
plt.scatter(house_features[:, 0], house_prices, color = 'blue', label='Actual Prices')  # Actual prices
plt.scatter(new_house[:, 0], predicted_price, color='red', label='Predicted Price', marker='x', s=100)  # Predicted price
plt.xlabel("House Size (m²)")
plt.ylabel("House Price ($)")
plt.title("Effect of House Size on Price")
plt.legend()
plt.show()
