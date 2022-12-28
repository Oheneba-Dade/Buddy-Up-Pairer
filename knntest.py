import pandas as pd
from sklearn.neighbors import NearestNeighbors


data_dir = "freshmen.csv"
data_dir2 = "older buddies.csv"
ClassData = pd.read_csv(data_dir)
BuddiesData = pd.read_csv(data_dir2)
student_names = ClassData["Name"].values
buddy_names = BuddiesData["Name"].values

ClassDataNums = pd.read_csv(data_dir)  # this stores the data without the names
ClassDataNums = ClassDataNums.drop(["Name"], axis=1)
BuddiesDataNums = pd.read_csv(data_dir2)  # this stores the data without the names
BuddiesDataNums = BuddiesDataNums.drop(["Name"], axis=1)


neigh = NearestNeighbors(n_neighbors=len(student_names))
neigh.fit(ClassDataNums)


def generate_buddies_list():
    """
    For each buddy, find the closest students to them and return a dictionary of the form {buddy:
    [(student, distance), (student, distance), ...]}
    :return: A dictionary of buddies.
    """
    buddy_dict = {}
    for buddy in buddy_names:
        buddy_index = BuddiesData[BuddiesData["Name"] == buddy].index[0]
        buddy_data = BuddiesDataNums.iloc[buddy_index]
        buddy_data = buddy_data.values.reshape(1, -1)
        distances, indices = neigh.kneighbors(buddy_data)
        buddies = []
        for i in range(len(indices[0])):
            buddies.append((student_names[indices[0][i]], distances[0][i]))
        buddy_dict[buddy] = buddies
    return buddy_dict


buddy_info = generate_buddies_list()
for i in buddy_info:
    print(buddy_info[i])
    print()
    print()
