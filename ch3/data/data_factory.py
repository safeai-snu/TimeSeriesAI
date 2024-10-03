from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom
}

def data_provider(args, flag):
    Data = Dataset_ETT_hour
    timeenc = 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 32
        freq = 'h'
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 32
        freq = 'h'
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = 32
        freq = 'h'

    data_set = Data(
        root_path='./dataset/',
        data_path='ETTh1.csv',
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features='M',
        target='OT',
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=10,
        drop_last=drop_last)
    return data_set, data_loader
