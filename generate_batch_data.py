import pandas as pd
import random
import numpy as np

# Simulate a college batch of 1,200 students
num_students = 1200
categories = ['normal', 'stress', 'anxiety', 'depression']
weights = [0.50, 0.30, 0.15, 0.05] # Realistic distribution

data = {
    'student_id': [f"STU-{1000+i}" for i in range(num_students)],
    'department': random.choices(['CS', 'IT', 'Mechanical', 'Civil'], k=num_students),
    'last_state': random.choices(categories, weights=weights, k=num_students),
    'avg_score': [round(random.uniform(1.0, 4.0), 2) for _ in range(num_students)]
}

df = pd.DataFrame(data)
df.to_csv("data/batch_wellness_report.csv", index=False)
print("Large-scale Batch Data generated in data/batch_wellness_report.csv")