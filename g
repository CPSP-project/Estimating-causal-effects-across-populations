

import csv

# Dataset 1: Popolazione osservata
observational_population = [
    ["id", "X1", "X2", "Z", "Y"],
    [1, 5.2, 1, 2.3, 10.1],
    [2, 4.8, 0, 1.8, 9.5],
    [3, 6.1, 1, 2.9, 11.0],
    [4, 5.5, 0, 2.1, 10.0],
    [5, 4.3, 1, 1.7, 8.8],
    [6, 5.9, 0, 2.5, 10.2],
    [7, 4.7, 1, 2.0, 9.3],
    [8, 6.3, 0, 3.0, 11.4],
    [9, 5.0, 1, 2.2, 9.9],
    [10, 5.7, 0, 2.4, 10.3],
]

with open("observational_population.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(observational_population)

# Dataset 2: Studio
study_dataset = [
    ["id", "X", "Z", "Y"],
    [1, 1, 2.1, 10.0],
    [2, 0, 1.9, 9.5],
    [3, 1, 2.8, 11.1],
    [4, 1, 2.0, 10.2],
    [5, 0, 1.7, 9.0],
    [6, 1, 2.3, 10.8],
    [7, 0, 2.0, 9.2],
    [8, 1, 3.1, 11.3],
    [9, 0, 2.2, 9.8],
    [10, 1, 2.5, 10.7],
]

with open("study_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(study_dataset)

# Dataset 3: Intervento nello studio
intervention_dataset = [
    ["id", "X", "Z"],
    [1, 1, 2.2],
    [2, 1, 2.0],
    [3, 1, 2.7],
    [4, 1, 2.1],
    [5, 1, 1.8],
    [6, 1, 2.4],
    [7, 1, 2.1],
    [8, 1, 3.0],
    [9, 1, 2.3],
    [10, 1, 2.6],
]

with open("intervention_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(intervention_dataset)

print("I tre file CSV sono stati creati esattamente come da specifiche!")
