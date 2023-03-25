import torch
if __name__ == '__main__':
    batch = [{"video":[0,1,2],"video_name":['name1','name2','name3','name4']},{"video":[0,1,2,3],"video_name":['name1','name2','name3','name4']},{"video":[0,1,2,3],"video_name":['name1','name2','name3','name4']}]
    all=[]
    a=[]
    b=[]
    list_batch=[]
    for i in range(len(batch)):
        for k,v in batch[i].items():
            if k == 'video':
                a.append(v)
            else:
                b.append(v)
            all.append(v)
    print("a:",a)
    print("b:",b)
    batch = list(zip(*(a)))
    print("a_zip:",batch)
    batch=list(zip(*(b)))
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            list_batch.append(batch[i][j])
    print("b_zip:",batch)
    print(list_batch)

    video_batch = list(zip(*(a)))
    video_name_batch = list(zip(*(b)))
    for i in range(len(video_batch)):
        for j in range(len(video_batch[i])):
            video_batch_list.append(video_batch[i][j].numpy())
    for i in range(len(video_name_batch)):
        for j in range(len(video_name_batch[i])):
            video_name_batch_list.append(video_name_batch[i][j])
    video = torch.tensor(video_batch_list, dtype=torch.int32)
    video_name = video_name_batch_list



def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    all = []
    a=[]
    b=[]
    video_batch_list=[]
    video_name_batch_list=[]
    for i in range(len(batch)):
        for k, v in batch[i].items():
            all.append(v)
            if k == 'video':
                a.append(v)
            else:
                b.append(v)
    video_batch = list(zip(*(a)))
    video_name_batch=list(zip(*(b)))
    for i in range(len(video_batch)):
        for j in range(len(video_batch[i])):
            video_batch_list.append(video_batch[i][j].numpy())
    for i in range(len(video_name_batch)):
        for j in range(len(video_name_batch[i])):
            video_name_batch_list.append(video_name_batch[i][j])
    video= torch.tensor(video_batch_list, dtype=torch.int32)
    video_name = video_name_batch_list
    del batch
    return video, video_name