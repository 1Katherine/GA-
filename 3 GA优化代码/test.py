import pandas as pd

# --------------------- 生成 gan-rs 初始种群 start -------------------
initpoint_path = './wordcount-100G-GAN.csv'
initsamples_df = pd.read_csv(initpoint_path)

def ganrs_samples_all():
    # 初始样本
    initsamples = initsamples_df.to_numpy()
    return initsamples

def ganrs_samples_odd():
    initsamples_odd = initsamples_df[initsamples_df.index % 2 == 0]
    initsamples = initsamples_odd.to_numpy()
    return initsamples

def ganrs_samples_even():
    initsamples_even = initsamples_df[initsamples_df.index % 2 == 1]
    initsamples = initsamples_even.to_numpy()
    return initsamples

def get_ganrs_samples(kind):
    if sample_kind == 'all':
        samples = ganrs_samples_all()
    if sample_kind == 'odd':
        samples = ganrs_samples_odd()
    if sample_kind == 'even':
        samples = ganrs_samples_even()
    return  samples

# --------------------- 生成 gan-rs 初始种群 end -------------------

if __name__ == '__main__':
    sample_kind = 'even'
    ganrs_samples = get_ganrs_samples(kind=sample_kind)
    print(ganrs_samples)
    print(ganrs_samples.shape)