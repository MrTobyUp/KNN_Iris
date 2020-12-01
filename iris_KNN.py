from collections import Counter
import math
import pandas as pd

def knn(data, query, k, choice_fn):
    distances = []

    # Den Euklidschen Abstand von jedem Punkt zum übergeben Eintrag berechnen
    for index, example in enumerate(data):
        distance = euclidean_distance(example[:-1], query) # beginning to last 

        # Die Entfernung und ein dazugehöriger Index wird in distance gespeichert
        distances.append((distance, index))

    # Das distance Array sortieren, damit man die kürzesten Abstände leicht bekommt
    sorted_distances = sorted(distances)

    # Die ersten K Einträge herausholen
    k_first_entries = sorted_distances[:k]

    # k_nearest_labels = [data[i][1] for distance, i in k_nearest_distances_and_indices]
    # Die Ausgabewerte der selektierten. k_first_entries herausholen
    k_nearest_labels = []
    for label in k_first_entries:
        # [-1] ist der letzte Eintrag im 2. dimensionalen (den Wert von dem uns das Ergebnis interessiert)
        k_nearest_labels.append(data[label[1]][-1])
    
    # Bei Regression = choice_fn=mean
    # Bei Klassification = choice_fn=mode
    # Die nächsten Nachbarn und das Ergebnis zurückgeben
    return k_first_entries, choice_fn(k_nearest_labels)


def mean(labels):
    return sum(labels) / len(labels)


def mode(labels):
    return Counter(labels).most_common(1)[0][0]


def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    # Für jeden eintrag den Abstand berechnen -> wichtig bei Mehrdimensionalen Arrays
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)

def load_csv(csv_path):
    dataframe = pd.read_csv(csv_path)

    # 0: petal_length
    # 1: petal_width
    # 2: flower_type -> #2 muss immer der gewünschte Output sein
    value_array = dataframe[["petal_length", "petal_width", "species"]].to_numpy()
    return value_array

def main():
    reg_data = load_csv("iris.csv")

    # Welcher Blumentyp hat die Dimensionen [petal_length=56, petal_width=24]
    reg_query = [25, 10]
    nearest_neighbors, prediction = knn(
        reg_data, reg_query, k=3, choice_fn=mode
    )
    # Bei Regression = choice_fn=mean
    # Bei Klassification = choice_fn=mode
    print("The nearest neigbours: {}".format(nearest_neighbors))
    print("Prediction: {}".format(prediction))

if __name__ == '__main__':
    main()