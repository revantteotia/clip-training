from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


def get_dataloader(config, dataset, is_train = True):
    
    if is_train:
        sampler = RandomSampler(dataset)
        batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    else:
        sampler = SequentialSampler(dataset)
        batch_size = config.per_gpu_eval_batch_size * max(1, config.n_gpu)

    dataloader = DataLoader(dataset, sampler=sampler, 
            batch_size=batch_size, num_workers=config.num_workers)

    return dataloader
