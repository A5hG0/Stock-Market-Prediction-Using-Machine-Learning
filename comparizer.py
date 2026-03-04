#Regression
"""             r2                                     rmse
    knn         0.3823606850050765                    318.1530666760614
    lr          0.9848396525376479                    49.81043013349531
    rfr         0.6836081517276146                    227.70957367550147
"""
#Classification
"""             precision          recall          f1-score              accuracy
    knn    0     0.70               0.75             0.72                70.9897610921501%
           1     0.73               0.67             0.70
           
    lr           0.76               0.73             0.75                75.0853242320819%
                 0.74               0.77             0.75
    
    rfc          0.67               0.76             0.71                75.68.94197952218431%
                0.72                0.62             0.67
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Regression Results
# -------------------------------
import matplotlib.pyplot as plt

models = ["KNN", "Linear Regression", "Random Forest"]
rmse = [318.15, 49.81, 227.70]

plt.figure(figsize=(8,5))
plt.bar(models, rmse)

plt.title("Regression Model Comparison (RMSE)")
plt.ylabel("RMSE (Lower is Better)")
plt.xlabel("Models")

plt.text(0, rmse[0]+5, "Higher error")
plt.text(1, rmse[1]+5, "Best model")
plt.text(2, rmse[2]+5, "Moderate error")

plt.show()


# -------------------------------
# Classification Results
# -------------------------------
models = ["KNN", "Logistic Regression", "Random Forest"]
accuracy = [70.98, 75.08, 75.68]

plt.figure(figsize=(8,5))
plt.bar(models, accuracy)

plt.title("Classification Model Comparison (Accuracy)")
plt.ylabel("Accuracy % (Higher is Better)")
plt.xlabel("Models")

plt.text(0, accuracy[0]+0.5, "Lower")
plt.text(1, accuracy[1]+0.5, "Better")
plt.text(2, accuracy[2]+0.5, "Best")

plt.show()