import os
from datasets import Dataset, DatasetDict
try:
    from config import train_data
except ImportError:
    import sys
    sys.path.append(sys.path[0] + '/..')
    from config import train_data

def split_dict_into_three(d):
    # Sort the dictionary by values and get the keys in sorted order
    sorted_keys = sorted(d, key=d.get)

    # Determine the size of each split
    total_keys = len(sorted_keys)
    part_size = total_keys // 3

    # Create the three parts
    lowest_keys = sorted_keys[:part_size]
    mid_keys = sorted_keys[part_size:2 * part_size]
    highest_keys = sorted_keys[2 * part_size:]

    # Handle any remaining keys due to integer division
    remainder = total_keys % 3
    if remainder > 0:
        highest_keys.extend(sorted_keys[-remainder:])

    return sorted(lowest_keys), sorted(mid_keys), sorted(highest_keys)

def split_curriculum(dataset):
    curriculum_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ratings_all.txt")
    ratings = {}
    with open(curriculum_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    lines = lines[:-1]
    if len(lines) != len(dataset):
        raise Exception("Dataset doesn't match ratings file")
    for line in lines:
        index, score = line.split(': ')
        ratings[int(index)] = float(score)

    high_indices, mid_indices, low_indices = split_dict_into_three(ratings)
    print(len(high_indices), len(mid_indices), len(low_indices))

    high_points = set()
    mid_points = set()
    low_points = set()
    for i, point in enumerate(dataset["text"]):
        if i % 1_000 == 0:
            print(i)
        if i in high_indices:
            high_points.add(point)
        elif i in mid_indices:
            mid_points.add(point)
        elif i in low_indices:
            low_points.add(point)

    splits = {}

    high_data = Dataset.from_dict({"text": list(high_points)})
    mid_data = Dataset.from_dict({"text": list(mid_points)})
    low_data = Dataset.from_dict({"text": list(low_points)})

    splits["high_data"] = high_data
    splits["mid_data"] = mid_data
    splits["low_data"] = low_data

    split_dataset = DatasetDict(splits)
    print("Split complete, saving datasets to disk")

    split_dataset.save_to_disk("split_dataset")

    # dataset = dataset.map(lambda data, idx: {'index': idx}, with_indices=True)
    #
    #
    # high_data = dataset.filter(lambda data: data['index'] in high_indices)
    # print("This should work now")
    # mid_data = dataset.filter(lambda data: data['index'] in mid_indices)
    # low_data = dataset.filter(lambda data: data['index'] in low_indices)

    # return high_data, mid_data, low_data

split_curriculum(train_data)