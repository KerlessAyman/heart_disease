def validate_input(inputs):
    ranges = {
        'age': (0, 100),
        'sex': (0, 1),
        'cp': (0, 3),
        'trestbps': (90, 200),
        'chol': (100, 600),
        'fbs': (0, 1),
        'restecg': (0, 2),
        'thalach': (60, 200),
        'exang': (0, 1),
        'oldpeak': (0.0, 6.9),
        'slope': (0, 2),
        'ca': (0, 3),
        'thal': (0, 3),
    }
    for key, value in inputs.items():
        if not (ranges[key][0] <= value <= ranges[key][1]):
            return False
    return True