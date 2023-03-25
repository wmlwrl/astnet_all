import torch

if __name__ == '__main__':
    a = []
    b = []
    batch = [{"video": [0, 1, 2], "video_name": ['name1', 'name2', 'name3', 'name4']},
             {"video": [0, 1, 2, 3], "video_name": ['name1', 'name2', 'name3', 'name4']}]
    for i in range(len(batch)):
        for k, v in batch[i].items():
            if k == 'video':
                a.append(batch[i].items())
            else:
                b.append(batch[i].items())
    print(a)


