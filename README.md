# Inspiration
We were curious about the complexity of airline operations and the challenges faced in scheduling and performance. With growing demand for air travel, the need to optimize resources while ensuring compliance with regulatory requirements is growing. We realized there was an opportunity to leverage data and predictive models to streamline operations. By predicting passenger numbers more accurately, we can significantly improve crew assignments and reduce operational costs, all while enhancing the overall passenger experience.

# What it does
Our model uses historical flight data combined with various predictors such as fare and distance traveled to accurately forecast the number of passengers for each flight. The model provides insights into crew requirements based on predicted passenger loads, making staffing more efficient.

# How we built it
We built the model using machine learning techniques, as well as pre-processing our data to better our results. For modeling, we used XGBoost to develop a regression model that predicts the number of passengers for each flight. We then linked this prediction to crew assignment optimization, allowing the model to determine the ideal number of crew members needed for each flight.

# Challenges we ran into
One major challenge we encountered was learning and fine-tuning XGBoost. Understanding how to optimize our parameters and handle different feature sets took some time. Another challenge was debugging our Streamlit code to successfully transfer our backend model into a usable, interactive front-end application.

# Accomplishments that we're proud of
We’re proud of the accuracy we achieved in predicting passenger numbers using our model. By carefully selecting and engineering features, we’ve developed a model that can provide actionable insights for cabin crew scheduling. We’re also proud of the efficient crew assignment system that can be implemented based on these predictions, offering real value to airline operations.

# What we learned
This project taught us the importance of feature selection and the details that can significantly impact model accuracy. We also gained a deeper understanding of what goes into properly running a large operation that impacts others. Finally, we learned that predictive models can always be improved with the right external data sources and thoughtful feature engineering, and this experience highlighted the potential of data-driven decision-making in real-world applications.

# What's next
Next, we aim to enhance the model by incorporating more external data, such as real-time weather conditions, flight delays, and economic indicators, to refine our predictions further. Also, we hope to implement our predictor model to a wider variety of airlines. We only focused on airlines that offered affordable airfare, so we would like to create accurate results that could be applicable to a wider scope of airlines.

# Technologies
Python: Pandas, NumPy, Sklearn, Matplotlib, Seaborn, XGBoost, Streamlit, Jupyter Notebook

# To Run
To run the program locally, type in streamlit run script.py in the terminal.
