from collections import Counter
import math
import pandas as pd

def knn(data, query, k, choice_fn):
    distances = []

    # Calculate the Euclidean distance from each point to the given entry
    for index, example in enumerate(data):
        distance = euclidean_distance(example[:-1], query) # beginning to last 

        # The distance and a corresponding index is stored in distance
        distances.append((distance, index))

    # Sort the distance array to get the shortest distances easily
    sorted_distances = sorted(distances)

    # Get the first K entries out
    k_first_entries = sorted_distances[:k]

    # k_nearest_labels = [data[i][1] for distance, i in k_nearest_distances_and_indices]
    # Get the output values of the selected. k_first_entries
    k_nearest_labels = []
    for label in k_first_entries:
        # [-1] is the last entry in the 2nd dimensional (the value of which we are interested in the result)
        k_nearest_labels.append(data[label[1]][-1])
    
    # With regression = choice_fn=mean
    # For classification = choice_fn=mode
    # Return the nearest neighbors and the result
    return k_first_entries, choice_fn(k_nearest_labels)


def mean(labels):
    return sum(labels) / len(labels)


def mode(labels):
    return Counter(labels).most_common(1)[0][0]


def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    # calculate the distance for each entry -> important for multidimensional arrays
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)

def load_csv(csv_path):
    dataframe = pd.read_csv(csv_path)

    # 0: petal_length
    # 1: petal_width
    # 2: flower_type -> #2 must always be the desired output
    value_array = dataframe[["petal_length", "petal_width", "species"]].to_numpy()
    return value_array

def main():
    reg_data = load_csv("iris.csv")

    # Which flower type has the dimensions [petal_length=56, petal_width=24]
    reg_query = [25, 10]
    nearest_neighbors, prediction = knn(
        reg_data, reg_query, k=3, choice_fn=mode
    )
    # With regression = choice_fn=mean
    # With classification = choice_fn=mode
    print("The nearest neigbours: {}".format(nearest_neighbors))
    print("Prediction: {}".format(prediction))

if __name__ == '__main__':
    main()
