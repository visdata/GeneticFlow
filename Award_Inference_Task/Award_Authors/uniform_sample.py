def uniform_sample(all_items, sample_num):
    all_num = len(all_items)
    if sample_num > all_num or sample_num <= 0:
        print("Invalid sample: ", sample_num, " from ", all_num)
        return None
    sampled_items=[]
    sample_interval = (float(all_num)/sample_num)
    current_sample_index = 0
    for i in range(sample_num):
        sampled_items.append(all_items[int(current_sample_index)])
        current_sample_index+=sample_interval
        
    return sampled_items