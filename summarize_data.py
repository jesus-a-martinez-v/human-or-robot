import pandas as pd

__bids_file_path = "./data/bids.csv"


def __get_summary_per_user_dataframe(save=False, path="./data/summary_per_bidder.csv"):
    def rename_columns(df, name_mapping):
        df.rename(columns=name_mapping, inplace=True)

    bids_dataframe = pd.read_csv(__bids_file_path)
    data_per_user = bids_dataframe.groupby(['bidder_id'])

    auctions_per_user = data_per_user['auction'].nunique().to_frame()
    bids_per_user = data_per_user['bid_id'].count().to_frame()
    countries_per_user = data_per_user['country'].nunique().to_frame()
    ips_per_user = data_per_user['ip'].nunique().to_frame()
    bids_per_auction_ratio_per_user = (bids_per_user['bid_id'] / auctions_per_user['auction']).to_frame()
    average_response_time_per_user = data_per_user['time'].apply(lambda x: x.diff().mean()).fillna(0).to_frame()

    summary_per_bidder = auctions_per_user.join(bids_per_user).join(countries_per_user).join(ips_per_user).join(
        bids_per_auction_ratio_per_user).join(average_response_time_per_user)

    rename_columns(summary_per_bidder, {'auction': 'number_of_auctions',
                                        'bid_id': 'number_of_bids',
                                        'country': 'number_of_countries',
                                        'ip': 'number_of_ips',
                                        0: 'bids_auction_ratio',
                                        'time': 'average_response_time'})

    if save:
        summary_per_bidder.to_csv(path, encoding='utf-8', index=False)

    return summary_per_bidder


def __merge_with_data_set(dataset, summary_dataset):
    return dataset.join(summary_dataset, how="left", on="bidder_id").fillna(0)


def summarize_data():
    summary = __get_summary_per_user_dataframe(save=True)
    __merge_with_data_set(pd.read_csv("./data/train.csv"), summary).to_csv("./data/merged_train.csv", encoding='utf-8',
                                                                           index=False)
    __merge_with_data_set(pd.read_csv("./data/test.csv"), summary).to_csv("./data/merged_test.csv", encoding='utf-8',
                                                                          index=False)


if __name__ == "__main__":
    summarize_data()
