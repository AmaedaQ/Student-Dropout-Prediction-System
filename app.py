import pickle
from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the best model, PCA, and scaler
with open('model/best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Load the saved PCA (we'll use this later)
with open('model/pca.pkl', 'rb') as file:
    pca = pickle.load(file)

# Initialize the Flask application
app = Flask(__name__)

# Initialize a StandardScaler (manually scale the features)
scaler = StandardScaler()

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Fetch the form data (user inputs)
        marital_status = int(request.form['marital_status'])
        application_mode = int(request.form['application_mode'])
        application_order = int(request.form['application_order'])
        course = int(request.form['course'])
        daytime_evening = int(request.form['attendance'])
        previous_qualification = int(request.form['previous_qualification'])
        mothers_qualification = int(request.form['mothers_qualification'])
        fathers_occupation = int(request.form['fathers_occupation'])
        displaced = int(request.form['displaced'])
        educational_special_needs = int(request.form['educational_special_needs'])
        debtor = int(request.form['debtor'])
        tuition_fees_up_to_date = int(request.form['tuition_fees_up_to_date'])
        gender = int(request.form['gender'])
        scholarship_holder = int(request.form['scholarship_holder'])
        age_at_enrollment = int(request.form['age_at_enrollment'])
        international = int(request.form['international'])

        # Curricular features (to be transformed using PCA)
        curricular_1st_sem_credited = int(request.form['curricular_units_1st_sem_credited'])
        curricular_1st_sem_enrolled = int(request.form['curricular_units_1st_sem_enrolled'])
        curricular_1st_sem_evaluations = int(request.form['curricular_units_1st_sem_evaluations'])
        curricular_1st_sem_without_evaluations = int(request.form['curricular_units_1st_sem_without_evaluations'])
        curricular_1st_sem_approved = int(request.form['curricular_units_1st_sem_approved'])
        curricular_1st_sem_grade = float(request.form['curricular_units_1st_sem_grade'])
        curricular_2nd_sem_credited = int(request.form['curricular_units_2nd_sem_credited'])
        curricular_2nd_sem_enrolled = int(request.form['curricular_units_2nd_sem_enrolled'])
        curricular_2nd_sem_evaluations = int(request.form['curricular_units_2nd_sem_evaluations'])
        curricular_2nd_sem_without_evaluations = int(request.form['curricular_units_2nd_sem_without_evaluations'])
        curricular_2nd_sem_approved = int(request.form['curricular_units_2nd_sem_approved'])
        curricular_2nd_sem_grade = float(request.form['curricular_units_2nd_sem_grade'])

        # Step 2: Prepare the input data array (matching the 17 expected features)
        input_data = np.array([
            marital_status, application_mode, application_order, course, daytime_evening,
            previous_qualification, mothers_qualification, fathers_occupation, displaced,
            educational_special_needs, debtor, tuition_fees_up_to_date, gender, scholarship_holder,
            age_at_enrollment, international,
            curricular_1st_sem_credited, curricular_1st_sem_enrolled, curricular_1st_sem_evaluations,
            curricular_1st_sem_without_evaluations, curricular_1st_sem_approved, curricular_1st_sem_grade,
            curricular_2nd_sem_credited, curricular_2nd_sem_enrolled, curricular_2nd_sem_evaluations,
            curricular_2nd_sem_without_evaluations, curricular_2nd_sem_approved, curricular_2nd_sem_grade
        ]).reshape(1, -1)

        # Step 3: Manually scale the input features (excluding PCA)
        features_to_scale = input_data[:, :16]  # Only scale the first 16 features
        scaled_features = scaler.fit_transform(features_to_scale)  # Fit and transform the features

        # Step 4: Apply PCA transformation to the curricular features (since PCA is trained on them)
        curricular_features = input_data[:, 16:]  # The last feature is PCA-related
        pca_transformed = pca.transform(curricular_features)  # Apply PCA

        # Combine the scaled features and PCA result
        final_input = np.hstack((scaled_features, pca_transformed))

        # Step 5: Make the prediction using the trained model
        prediction = best_model.predict(final_input)

        # Step 6: Interpret the prediction result
        # Since your model only predicts Dropout (True/False), you need to map that:
        if prediction == 1:
            result = 'Dropout'
            risk_level = 'High risk of dropping out based on current academic and personal factors. We recommend reviewing study habits, seeking support services, and exploring available resources to stay on track.'
        else:
            result = 'Not Dropout'  # Assuming 'False' maps to 'Enrolled'
            risk_level = 'Currently enrolled. Stay proactive with your studies to ensure continued success. Consider accessing additional resources for academic support if needed.'

        # Return the result to the user
        return render_template('prediction.html', result=result, risk_level=risk_level)

    return render_template('prediction.html')


# Route for the results page
@app.route('/results')
def results():
    return render_template('results.html')  # Customize it later

# Route for the about page
@app.route('/about')
def about():
    return render_template('about.html')  # Customize it later


if __name__ == '__main__':
    app.run(debug=True)
