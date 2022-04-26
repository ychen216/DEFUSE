import copy
import os

import pandas as pd
import numpy as np
import pickle

from utils import parse_float_arg

SECONDS_A_DAY = 60*60*24
SECONDS_AN_HOUR = 60*60
SECONDS_DELAY_NORM = 1
SECONDS_FSIW_NORM = SECONDS_A_DAY*5
# the input of neural network should be normalized


def get_data_df(params):
    df = pd.read_csv(params["data_path"], sep="\t", header=None)
    click_ts = df[df.columns[0]].to_numpy()
    pay_ts = df[df.columns[1]].fillna(-1).to_numpy()

    df = df[df.columns[2:]]
    for c in df.columns[8:]:
        df[c] = df[c].fillna("")
        df[c] = df[c].astype(str)
    for c in df.columns[:8]:
        df[c] = df[c].fillna(-1)
        df[c] = (df[c] - df[c].min())/(df[c].max() - df[c].min())
    df.columns = [str(i) for i in range(17)]
    df.reset_index(inplace=True)
    return df, click_ts, pay_ts


class DataDF(object):

    def __init__(self, features, click_ts, pay_ts, sample_ts=None, labels=None, delay_labels=None, inw_labels=None, attr_win = None):
        self.x = features.copy(deep=True)
        self.click_ts = copy.deepcopy(click_ts)
        self.pay_ts = copy.deepcopy(pay_ts)
        self.delay_labels = delay_labels
        self.inw_labels = inw_labels
        if sample_ts is not None:
            self.sample_ts = copy.deepcopy(sample_ts)
        else:
            self.sample_ts = copy.deepcopy(click_ts)
        if labels is not None:
            self.labels = copy.deepcopy(labels)
        else:
            if attr_win is not None:
                self.labels = (np.logical_and(pay_ts > 0, pay_ts - click_ts < attr_win)).astype(np.int32)
            else:
                self.labels = (pay_ts > 0).astype(np.int32)
        if self.delay_labels is None:
            self.delay_labels = self.labels
        if self.inw_labels is None:
            self.inw_labels = self.labels


    def sub_days(self, start_day, end_day):
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)
        return DataDF(self.x.iloc[mask],
                      self.click_ts[mask],
                      self.pay_ts[mask],
                      self.sample_ts[mask],
                      self.labels[mask])

    def sub_days_v2(self, start_day, end_day, cut_size, attr_win = None):
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)

        if attr_win is not None:
            diff = self.pay_ts - self.click_ts
            delay_mask = np.logical_and(self.pay_ts > 0, np.logical_and(diff > cut_size, diff < attr_win))
            attr_mask = np.logical_and(self.pay_ts > 0, diff < attr_win)
            self.labels[~attr_mask] = 0
        else:
            delay_mask = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts > cut_size)
        

        labels = copy.deepcopy(self.labels)
        labels[self.pay_ts > end_ts] = 0
        delay_labels = copy.deepcopy(self.labels)
        delay_labels[~delay_mask] = 0
        return DataDF(self.x.iloc[mask],
                      self.click_ts[mask],
                      self.pay_ts[mask],
                      self.sample_ts[mask],
                      labels[mask],
                      delay_labels[mask])

    def sub_hours(self, start_hour, end_hour):
        start_ts = start_hour*SECONDS_AN_HOUR
        end_ts = end_hour*SECONDS_AN_HOUR
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)
        if self.delay_labels is not None:
            return DataDF(self.x.iloc[mask],
                        self.click_ts[mask],
                        self.pay_ts[mask],
                        self.sample_ts[mask],
                        self.labels[mask],
                        self.delay_labels[mask],
                        self.inw_labels[mask])
        return DataDF(self.x.iloc[mask],
                    self.click_ts[mask],
                    self.pay_ts[mask],
                    self.sample_ts[mask],
                    self.labels[mask])

    def mask_rn(self, attr_win):
        if attr_win is not None:
            mask = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts <= attr_win) # pay in attr_win
        else:
            mask = self.pay_ts > 0
        labels = copy.deepcopy(self.labels)
        labels[~mask] = 0
        return DataDF(self.x,
                    self.click_ts,
                    self.pay_ts,
                    self.sample_ts,
                    labels)

    def mask_rn_v2(self, attr_win = None):
        if attr_win is not None:
            mask = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts < attr_win)
        else:
            mask = self.pay_ts > 0
        return DataDF(self.x[mask],
                    self.click_ts[mask],
                    self.pay_ts[mask],
                    self.sample_ts[mask],
                    self.labels[mask])


    def construct_dp_data_v1(self, ob_win):
        mask = self.pay_ts - self.click_ts > ob_win
        x = pd.concat(
            (self.x.iloc[~mask].copy(deep=True), self.x.iloc[mask].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts[~mask], self.pay_ts[mask]], axis=0)
        click_ts = np.concatenate(
            [self.click_ts[~mask], self.click_ts[mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts[~mask], self.pay_ts[mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        labels[~mask] = 0
        labels = np.concatenate([labels[~mask], labels[mask]], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])


    def construct_tn_dp_data_v1(self, cut_size, cut_day_sec):
        mask = self.pay_ts - self.click_ts > cut_day_sec
        inw_posmask = np.logical_and(self.pay_ts - self.click_ts <= cut_size, self.pay_ts > 0)
        delay_mask = np.logical_and(self.pay_ts - self.click_ts > cut_size, \
            self.pay_ts - self.click_ts <= cut_day_sec)
        x = pd.concat(
            (self.x.copy(deep=True), (self.x.copy(deep=True)[mask])))
        sample_ts = np.concatenate(
            [self.click_ts, self.pay_ts[mask]], axis=0)  # negative samples can only be used after cut_size hours
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        labels[mask] = 0
        inw_labels = copy.deepcopy(self.labels)
        inw_labels[~inw_posmask] = 0
        delay_labels = copy.deepcopy(self.labels)
        delay_labels[~delay_mask] = 0

        # insert delayed positives
        labels = np.concatenate([labels, np.ones((np.sum(mask),))], axis=0)
        delay_labels = np.concatenate([delay_labels, np.ones((np.sum(mask),))], axis=0)
        inw_labels = np.concatenate([inw_labels, np.zeros((np.sum(mask),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx],
                      delay_labels[idx],
                      inw_labels[idx])


    def add_fake_neg(self, attr_win = None):
        pos_mask = np.logical_and(self.pay_ts > 0, self.labels)
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[pos_mask].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts, self.pay_ts[pos_mask]], axis=0)
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[pos_mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[pos_mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        labels[pos_mask] = 0
        labels = np.concatenate([labels, np.ones((np.sum(pos_mask),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def only_pos(self):
        pos_mask = self.pay_ts > 0
        print(np.mean(pos_mask))
        print(self.pay_ts[pos_mask].shape)
        return DataDF(self.x.iloc[pos_mask],
                      self.click_ts[pos_mask],
                      self.pay_ts[pos_mask],
                      self.sample_ts[pos_mask],
                      self.labels[pos_mask])

    def to_tn(self):
        mask = np.logical_or(self.pay_ts < 0, self.pay_ts -
                             self.click_ts > SECONDS_AN_HOUR)
        x = self.x.iloc[mask]
        sample_ts = self.sample_ts[mask]
        click_ts = self.click_ts[mask]
        pay_ts = self.pay_ts[mask]
        label = pay_ts < 0
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    def to_dp(self):
        x = self.x
        sample_ts = self.sample_ts
        click_ts = self.click_ts
        pay_ts = self.pay_ts
        label = pay_ts - click_ts > SECONDS_AN_HOUR
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    
    def add_inw_outw_delay_positive(self, cut_size):
        mask = np.logical_and(self.pay_ts - self.click_ts > cut_size, self.labels)
        x = pd.concat((self.x.copy(deep=True), self.x.iloc[mask].copy(deep=True)))
        sample_ts = np.concatenate([self.click_ts, self.pay_ts[mask]], axis=0)
        click_ts = np.concatenate([self.click_ts, self.click_ts[mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        labels[mask] = 0  # fake negatives
        # insert delayed positives
        delay_labels = np.concatenate([np.zeros(labels.shape[0]), np.ones((np.sum(mask),))], axis=0)
        inw_labels = 1 - delay_labels
        labels = np.concatenate([labels, np.ones((np.sum(mask),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx],
                      delay_labels[idx],
                      inw_labels[idx])

    def form_vanilla(self, cut_size):
        mask = np.logical_and(self.pay_ts - self.click_ts > cut_size, self.labels)
        sample_ts = self.click_ts
        labels = copy.deepcopy(self.labels)
        labels[mask] = 0  # fake negatives
        idx = list(range(sample_ts.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(self.x.iloc[idx],
                      self.click_ts[idx],
                      self.pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def add_esdfm_cut_fake_neg(self, cut_size):
        mask = np.logical_and(self.pay_ts - self.click_ts > cut_size, self.labels)
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[mask].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts, self.pay_ts[mask]], axis=0)  # negative samples can only be used after cut_size hours
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        labels[mask] = 0  # fake negatives
        # insert delayed positives
        labels = np.concatenate([labels, np.ones((np.sum(mask),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def add_esdfm_cut_fake_neg_v1(self, cut_size):
        mask = np.logical_and(self.pay_ts - self.click_ts > cut_size, self.labels)
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[mask].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts, self.pay_ts[mask]], axis=0)  # negative samples can only be used after cut_size hours
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        delay_labels = copy.deepcopy(self.labels)
        # labels[mask] = 0  # fake negatives
        # insert delayed positives
        labels = np.concatenate([labels, np.ones((np.sum(mask),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def add_esdfm_cut_fake_neg_v2(self, cut_size):
        mask = np.logical_and(self.pay_ts - self.click_ts > cut_size, self.labels)
        inw_posmask = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts <= cut_size)
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[mask].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts, self.pay_ts[mask]], axis=0)  # negative samples can only be used after cut_size hours
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        delay_labels = copy.deepcopy(self.labels)
        inw_labels = copy.deepcopy(self.labels)
        labels[mask] = 0  # fake negatives
        delay_labels[~mask] = 0
        inw_labels[~inw_posmask] = 0
        # insert delayed positives
        labels = np.concatenate([labels, np.ones((np.sum(mask),))], axis=0)
        delay_labels = np.concatenate([delay_labels, np.zeros((np.sum(mask),))], axis=0)
        inw_labels = np.concatenate([inw_labels, np.zeros((np.sum(mask),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx],
                      delay_labels[idx],
                      inw_labels[idx])

    def add_defer_duplicate_samples(self, cut_size, attr_win):
        inw_mask = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts <= cut_size) # pay in window
        label_mask = np.logical_and(self.pay_ts > 0, self.labels)
        df1 = self.x.copy(deep=True) # observe data
        df2 = self.x.copy(deep=True) # duplicate data
        x = pd.concat([df1[inw_mask], df1[~inw_mask], df2[label_mask], df2[~label_mask]])
        sample_ts = np.concatenate(
            [self.click_ts[inw_mask], self.click_ts[~inw_mask], 
            self.pay_ts[label_mask], self.click_ts[~label_mask] + attr_win], axis=0)
        click_ts = np.concatenate([self.click_ts[inw_mask], self.click_ts[~inw_mask], \
            self.click_ts[label_mask], self.click_ts[~label_mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts[inw_mask], self.pay_ts[~inw_mask], \
            self.pay_ts[label_mask], self.pay_ts[~label_mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        # insert delayed positives
        labels = np.concatenate([np.ones((np.sum(inw_mask),)), np.zeros((np.sum(~inw_mask),)), \
            labels[label_mask], labels[~label_mask]], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def add_defer_duplicate_samples_oracle(self, cut_size, attr_win):
        inw_mask = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts <= cut_size) # pay in window
        label_mask = self.pay_ts > 0
        df1 = self.x.copy(deep=True) # observe data
        df2 = self.x.copy(deep=True) # duplicate data
        x = pd.concat([df1[inw_mask], df1[~inw_mask], df2[label_mask], df2[~label_mask]])
        sample_ts = np.concatenate(
            [self.pay_ts[inw_mask], self.click_ts[~inw_mask], 
            self.pay_ts[label_mask], self.click_ts[~label_mask]], axis=0)
        click_ts = np.concatenate([self.click_ts[inw_mask], self.click_ts[~inw_mask], \
            self.click_ts[label_mask], self.click_ts[~label_mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts[inw_mask], self.pay_ts[~inw_mask], \
            self.pay_ts[label_mask], self.pay_ts[~label_mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        # insert delayed positives
        labels = np.concatenate([np.ones((np.sum(inw_mask),)), np.zeros((np.sum(~inw_mask),)), \
            labels[label_mask], labels[~label_mask]], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def add_defer_duplicate_samples_oracle_v2(self, cut_size, attr_win):
        inw_mask = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts <= cut_size) # pay in window
        label_mask = np.logical_and(self.pay_ts > 0, self.labels)
        #x = pd.concat([self.x.copy(deep=True), self.x.copy(deep=True)])
        df1 = self.x.copy(deep=True) # observe data
        df2 = self.x.copy(deep=True) # duplicate data
        x = pd.concat([df1[inw_mask], df1[~inw_mask], df2[label_mask], df2[~label_mask]])
        sample_ts = np.concatenate(
            [self.click_ts[inw_mask], self.click_ts[~inw_mask], 
            self.click_ts[label_mask], self.click_ts[~label_mask]], axis=0)
        click_ts = np.concatenate([self.click_ts[inw_mask], self.click_ts[~inw_mask], \
            self.click_ts[label_mask], self.click_ts[~label_mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts[inw_mask], self.pay_ts[~inw_mask], \
            self.pay_ts[label_mask], self.pay_ts[~label_mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        # insert delayed positives
        labels = np.concatenate([np.ones((np.sum(inw_mask),)), np.zeros((np.sum(~inw_mask),)), \
            labels[label_mask], labels[~label_mask]], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def add_defer_duplicate_samples_v2(self, cut_size, attr_win):
        mask = self.pay_ts - self.click_ts > cut_size # dp
        # rn
        pos_mask2 = np.logical_or((self.pay_ts - self.click_ts > attr_win), (self.pay_ts <= 0)) 
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[mask].copy(deep=True), self.x.iloc[pos_mask2].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts+cut_size, self.pay_ts[mask], self.click_ts[pos_mask2]+attr_win], axis=0)  # negative samples can only be used after cut_size hours
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[mask], self.click_ts[pos_mask2]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask], self.pay_ts[pos_mask2]], axis=0)
        labels = copy.deepcopy(self.labels)
        labels[mask] = 0  # fake negatives
        # insert delayed positives
        labels = np.concatenate([labels, np.ones((np.sum(mask),)), np.zeros((np.sum(pos_mask2),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])


    def to_dfm_tune(self, cut_ts):
        label = np.logical_and(self.pay_ts > 0, self.pay_ts < cut_ts)
        delay = np.reshape(cut_ts - self.click_ts, (-1, 1))/SECONDS_DELAY_NORM
        labels = np.concatenate([np.reshape(label, (-1, 1)), delay], axis=1)
        return DataDF(self.x,
                      self.click_ts,
                      self.pay_ts,
                      self.sample_ts,
                      labels)

    def shuffle(self):
        idx = list(range(self.x.shape[0]))
        np.random.shuffle(idx)
        return DataDF(self.x.iloc[idx],
                      self.click_ts[idx],
                      self.pay_ts[idx],
                      self.sample_ts[idx],
                      self.labels[idx],
                      self.delay_labels[idx],
                      self.inw_labels[idx])


def get_criteo_dataset_stream(params):
    name = params["dataset"]
    print("loading datasest {}".format(name))
    cache_path = os.path.join(
        params["data_cache_path"], "{}.pkl".format(name))
    if params["data_cache_path"] != "None" and os.path.isfile(cache_path):
        print("cache_path {}".format(cache_path))
        print("\nloading from dataset cache")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        train_stream = data["train"]
        test_stream = data["test"]
    else:
        print("\ncan't load from cache, building dataset")
        df, click_ts, pay_ts = get_data_df(params)
        if name == "last_30_1d_train_test_oracle":
            data = DataDF(df, click_ts, pay_ts, attr_win=SECONDS_A_DAY)
            train_data = data.sub_days(30, 60)
            test_data = data.sub_days(30, 60)
            train_stream = []
            test_stream = []
            for tr in range(30*24, 59*24+23):
                train_hour = train_data.sub_hours(tr, tr+1)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels,
                                     "delay_labels": train_hour.labels,
                                     "inw_labels": train_hour.labels})
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels,
                                     "delay_labels": test_hour.labels,
                                     "inw_labels": test_hour.labels})
        elif name == "last_30_train_test_oracle":
            data = DataDF(df, click_ts, pay_ts)
            train_data = data.sub_days(30, 60)
            test_data = data.sub_days(30, 60)
            train_stream = []
            test_stream = []
            for tr in range(30*24, 59*24+23):
                train_hour = train_data.sub_hours(tr, tr+1)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels,
                                     "delay_labels": train_hour.labels,
                                     "inw_labels": train_hour.labels})
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})
        elif name == "last_30_train_test_dfm":
            data = DataDF(df, click_ts, pay_ts)
            train_data = data.sub_days(30, 60)
            test_data = data.sub_days(30, 60)
            train_stream = []
            test_stream = []
            for tr in range(30*24, 59*24+23):
                cut_ts = (tr+1)*SECONDS_AN_HOUR
                train_hour = train_data.sub_hours(tr, tr+1).to_dfm_tune(cut_ts)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels,
                                     "delay_labels": train_hour.labels,
                                     "inw_labels": train_hour.labels})
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})
        elif "last_30_train_test_bidefuse" in name:
            cut_hour = parse_float_arg(name, "cut_hour")
            print("cut_hour {}".format(cut_hour))
            ob_win = cut_hour*SECONDS_AN_HOUR
            if '1d' in name:
                data = DataDF(df, click_ts, pay_ts, attr_win=SECONDS_A_DAY)
            else:
                data = DataDF(df, click_ts, pay_ts)
            train_data = data.sub_days(0, 60).add_inw_outw_delay_positive(ob_win)
            test_data = data.sub_days(30, 60)
            train_stream = []
            test_stream = []
            for tr in range(30*24, 59*24+23):
                if 'shuffle' in name:
                    train_hour = train_data.sub_hours(tr, tr+1).shuffle()
                else:
                    train_hour = train_data.sub_hours(tr, tr+1) # hourly. ordered better than shuffle
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels,
                                     "delay_labels": train_hour.delay_labels,
                                     "inw_labels": train_hour.inw_labels})
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                delay_labels = test_hour.pay_ts - test_hour.click_ts > ob_win
                inw_labels = np.logical_and(test_hour.pay_ts > 0, \
                    test_hour.pay_ts - test_hour.click_ts <= ob_win)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels,
                                    "delay_labels": delay_labels,
                                    "inw_labels": inw_labels})
        elif "last_30_train_test_vanilla" in name:
            cut_hour = parse_float_arg(name, "cut_hour")
            print("cut_hour {}".format(cut_hour))
            ob_win = cut_hour*SECONDS_AN_HOUR
            if '1d' in name:
                data = DataDF(df, click_ts, pay_ts, attr_win = SECONDS_A_DAY)
            else:
                data = DataDF(df, click_ts, pay_ts)
            train_data = data.sub_days(0, 60).form_vanilla(ob_win)
            test_data = data.sub_days(30, 60)
            train_stream = []
            test_stream = []
            for tr in range(30*24, 59*24+23):
                if 'shuffle' in name:
                    train_hour = train_data.sub_hours(tr, tr+1).shuffle()
                else:
                    train_hour = train_data.sub_hours(tr, tr+1) # hourly. ordered better than shuffle
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels,
                                     "delay_labels": train_hour.delay_labels,
                                     "inw_labels": train_hour.inw_labels})
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})
        elif "last_30_train_test_esdfm" in name:
            cut_hour = parse_float_arg(name, "cut_hour")
            print("cut_hour {}".format(cut_hour))
            ob_win = cut_hour*SECONDS_AN_HOUR
            if '1d' in name:
                data = DataDF(df, click_ts, pay_ts, attr_win = SECONDS_A_DAY)
            else:
                data = DataDF(df, click_ts, pay_ts)
            if 'oracle' in name:
                print('using oracle: {}'.format(name))
                train_data = data.sub_days(0, 60).add_esdfm_cut_fake_neg_v2(ob_win)
            else:
                train_data = data.sub_days(0, 60).add_esdfm_cut_fake_neg(ob_win)
            test_data = data.sub_days(30, 60)
            train_stream = []
            test_stream = []
            for tr in range(30*24, 59*24+23):
                if 'shuffle' in name:
                    train_hour = train_data.sub_hours(tr, tr+1).shuffle()
                else:
                    train_hour = train_data.sub_hours(tr, tr+1) # hourly. ordered better than shuffle
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels,
                                     "delay_labels": train_hour.delay_labels,
                                     "inw_labels": train_hour.inw_labels})
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})
        elif "last_30_train_test_defer" in name:
            cut_hour = parse_float_arg(name, "cut_hour")
            attr_day = parse_float_arg(name, "attr_day")
            print("cut_hour {}".format(cut_hour))
            print("attr_day {}".format(attr_day))
            ob_win = cut_hour*SECONDS_AN_HOUR
            attr_win = attr_day*SECONDS_A_DAY
            data = DataDF(df, click_ts, pay_ts, attr_win=attr_win)
            if 'oracle' in name:
                train_data = data.sub_days(0, 60).add_defer_duplicate_samples_oracle_v2(ob_win, attr_win)
                #train_data = data.sub_days(0, 60).add_defer_duplicate_samples_oracle(ob_win, attr_win)
                # train_data = data.sub_days(30, 60)
            else:
                train_data = data.sub_days(0, 60).add_defer_duplicate_samples(ob_win, attr_win)
            test_data = data.sub_days(30, 60)
            valid_stream = []
            train_stream = []
            test_stream = []
            for tr in range(30*24, 59*24+23):
                train_hour = train_data.sub_hours(tr, tr+1)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels,
                                     "delay_labels": train_hour.delay_labels,
                                     "inw_labels": train_hour.inw_labels})
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})
        elif "last_30_train_test_fnw" in name:
            if '1d' in name:
                data = DataDF(df, click_ts, pay_ts, attr_win = SECONDS_A_DAY)
            else:
                data = DataDF(df, click_ts, pay_ts)
            train_data = data.sub_days(0, 60).add_fake_neg()
            test_data = data.sub_days(30, 60)
            train_stream = []
            test_stream = []
            for tr in range(30*24, 59*24+23):
                train_hour = train_data.sub_hours(tr, tr+1)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels,
                                     "delay_labels": train_hour.delay_labels,
                                     "inw_labels": train_hour.inw_labels})
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})
        else:
            raise NotImplementedError("{} data does not exist".format(name))
    if params["data_cache_path"] != "None":
        with open(cache_path, "wb") as f:
            pickle.dump({"train": train_stream, "test": test_stream}, f)
    return train_stream, test_stream


def get_criteo_dataset(params):
    name = params["dataset"]
    print("loading datasest {}".format(name))
    cache_path = os.path.join(
        params["data_cache_path"], "{}.pkl".format(name))
    if params["data_cache_path"] != "None" and os.path.isfile(cache_path):
        print("cache_path {}".format(cache_path))
        print("\nloading from dataset cache")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        train_data = data["train"]
        test_data = data["test"]
    else:
        print("\nbuilding dataset")
        df, click_ts, pay_ts = get_data_df(params)
        if '1d' in name:
            data = DataDF(df, click_ts, pay_ts, attr_win=SECONDS_A_DAY)
        else:
            data = DataDF(df, click_ts, pay_ts)
        if name == "baseline_prtrain":
            train_data = data.sub_days(0, 30).shuffle()
            mask = train_data.pay_ts < 0
            train_data.pay_ts[mask] = 30 * \
                SECONDS_A_DAY + train_data.click_ts[mask]
            test_data = data.sub_days(30, 60)
        elif "baseline_pretrain_v2" in name:
            cut_hour = parse_float_arg(name, "cut_hour")
            ob_win = int(SECONDS_AN_HOUR*cut_hour)
            train_data = data.sub_days_v2(0, 30, ob_win).shuffle()
            mask = train_data.pay_ts < 0
            train_data.pay_ts[mask] = 30 * \
                SECONDS_A_DAY + train_data.click_ts[mask]
            test_data = data.sub_days(30, 60)
        elif "baseline_pretrain_1d" in name:
            cut_hour = parse_float_arg(name, "cut_hour")
            ob_win = int(SECONDS_AN_HOUR*cut_hour)
            attr_win = SECONDS_A_DAY
            train_data = data.sub_days_v2(0, 30, ob_win, attr_win).shuffle()
            mask = train_data.pay_ts < 0
            train_data.pay_ts[mask] = 30 * \
                SECONDS_A_DAY + train_data.click_ts[mask]
            test_data = data.sub_days(30, 60)
        elif name == "dfm_prtrain":
            train_data = data.sub_days(0, 30).shuffle()
            train_data.pay_ts[train_data.pay_ts < 0] = SECONDS_A_DAY*30
            delay = np.reshape(train_data.pay_ts -
                               train_data.click_ts, (-1, 1))/SECONDS_DELAY_NORM
            train_data.labels = np.reshape(train_data.labels, (-1, 1))
            train_data.labels = np.concatenate(
                [train_data.labels, delay], axis=1)
            test_data = data.sub_days(30, 60)
        elif "tn_dp_mask30d_pretrain_1d" in name:
            print("preprocess mask30d")
            cut_hour = parse_float_arg(name, "cut_hour")
            ob_win = int(SECONDS_AN_HOUR*cut_hour)
            mask_sec = int(SECONDS_A_DAY*30)
            attr_win = SECONDS_A_DAY
            train_data = data.sub_days(0, 30).shuffle()
            train_diff = train_data.pay_ts - train_data.click_ts
            train_label_tn = np.reshape(np.logical_or(train_data.pay_ts < 0, \
                np.logical_or(train_data.pay_ts > mask_sec, train_diff > attr_win)), (-1, 1))
            train_label_dp = np.reshape(np.logical_and(train_data.pay_ts < mask_sec,
                np.logical_and(train_diff > ob_win, train_diff < attr_win)), (-1, 1))
            train_label = np.reshape(np.logical_and(train_data.pay_ts < mask_sec, \
                np.logical_and(train_data.pay_ts > 0, train_diff < attr_win)), (-1, 1))
            train_data.labels = np.concatenate(
                [train_label_tn, train_label_dp, train_label], axis=1)
            test_data = data.sub_days(30, 60)
            test_label_tn = np.reshape(test_data.pay_ts < 0, (-1, 1))
            test_label_dp = np.reshape(
                test_data.pay_ts - test_data.click_ts > ob_win, (-1, 1))
            test_label = np.reshape(test_data.pay_ts > 0, (-1, 1))
            test_data.labels = np.concatenate(
                [test_label_tn, test_label_dp, test_label], axis=1)
        elif "tn_dp_mask30d_pretrain" in name:
            print("preprocess mask30d")
            cut_hour = parse_float_arg(name, "cut_hour")
            ob_win = int(SECONDS_AN_HOUR*cut_hour)
            mask_sec = int(SECONDS_A_DAY*30)
            train_data = data.sub_days(0, 30).shuffle()
            train_label_tn = np.reshape(np.logical_or(train_data.pay_ts < 0, \
                train_data.pay_ts > mask_sec), (-1, 1))
            train_label_dp = np.reshape(np.logical_and(train_data.pay_ts < mask_sec,
                train_data.pay_ts - train_data.click_ts > ob_win), (-1, 1))
            train_label = np.reshape(train_data.pay_ts > 0, (-1, 1))
            train_data.labels = np.concatenate(
                [train_label_tn, train_label_dp, train_label], axis=1)
            test_data = data.sub_days(30, 60)
            test_label_tn = np.reshape(test_data.pay_ts < 0, (-1, 1))
            test_label_dp = np.reshape(
                test_data.pay_ts - test_data.click_ts > ob_win, (-1, 1))
            test_label = np.reshape(test_data.pay_ts > 0, (-1, 1))
            test_data.labels = np.concatenate(
                [test_label_tn, test_label_dp, test_label], axis=1)
        elif "bidefuse_pretrain" in name:
            cut_hour = parse_float_arg(name, "cut_hour")
            ob_win = int(SECONDS_AN_HOUR*cut_hour)
            print('ob_win: {}'.format(cut_hour))
            train_data = data.sub_days_v2(0, 30, ob_win).shuffle()
            train_label_inw = np.reshape(train_data.labels - train_data.delay_labels, (-1, 1))
            train_label_outw = np.reshape(train_data.delay_labels, (-1, 1))
            train_label = np.reshape(train_data.labels > 0, (-1, 1))
            train_data.labels = np.concatenate([train_label_inw, train_label_outw, train_label], axis=1)
            test_data = data.sub_days(30, 60)
            test_label_inw = np.reshape(test_data.labels - test_data.delay_labels, (-1, 1))
            test_label_outw = np.reshape(test_data.delay_labels, (-1, 1))
            test_label = np.reshape(test_data.labels > 0, (-1, 1))
            test_data.labels = np.concatenate([test_label_inw, test_label_outw, test_label], axis=1)
        elif "tn_dp_pretrain" in name:
            cut_hour = parse_float_arg(name, "cut_hour")
            ob_win = int(SECONDS_AN_HOUR*cut_hour)
            train_data = data.sub_days_v2(0, 60, ob_win).shuffle()
            train_label_tn = np.reshape(train_data.pay_ts < 0, (-1, 1))
            train_label_dp = np.reshape(
                train_data.pay_ts - train_data.click_ts > ob_win, (-1, 1))
            train_label = np.reshape(train_data.pay_ts > 0, (-1, 1))
            train_data.labels = np.concatenate(
                [train_label_tn, train_label_dp, train_label], axis=1)
            test_data = data.sub_days(30, 60)
            test_label_tn = np.reshape(test_data.pay_ts < 0, (-1, 1))
            test_label_dp = np.reshape(
                test_data.pay_ts - test_data.click_ts > ob_win, (-1, 1))
            test_label = np.reshape(test_data.pay_ts > 0, (-1, 1))
            test_data.labels = np.concatenate(
                [test_label_tn, test_label_dp, test_label], axis=1)
        elif "dp_pretrain" in name:
            cut_hour = parse_float_arg(name, "cut_hour")
            ob_win = int(SECONDS_AN_HOUR*cut_hour)
            train_data = data.sub_days(0, 30).mask_rn_v2().shuffle()
            train_label_dp = np.reshape(
                train_data.pay_ts - train_data.click_ts > ob_win, (-1, 1))
            train_label = np.reshape(
                train_data.pay_ts > 0, (-1, 1))
            train_data.labels = np.concatenate([train_label_dp], axis=1)
            test_data = data.sub_days(30, 60).mask_rn_v2()
            print("len of test_data : {}".format(len(test_data.labels)))
            test_label_dp = np.reshape(
                test_data.pay_ts - test_data.click_ts > ob_win, (-1, 1))
            test_label = np.reshape(test_data.pay_ts > 0, (-1, 1))
            test_data.labels = np.concatenate(
                [test_label_dp, test_label], axis=1)
        elif "dp_v2_1d_pretrain" in name:
            cut_hour = parse_float_arg(name, "cut_hour")
            ob_win = int(SECONDS_AN_HOUR*cut_hour)
            attr_win = SECONDS_A_DAY
            train_data = data.sub_days(0, 30).mask_rn_v2().shuffle()
            print("len of train_data : {}".format(len(train_data.labels)))
            train_diff = train_data.pay_ts - train_data.click_ts
            train_label_dp = np.reshape(
                np.logical_and(train_diff > ob_win, train_diff < attr_win), (-1, 1))
            train_label = np.reshape(np.logical_and(train_data.pay_ts > 0, \
                train_diff < attr_win), (-1, 1))
            train_data.labels = np.concatenate([train_label_dp], axis=1)
            test_data = data.sub_days(30, 60).mask_rn_v2()
            print("len of test_data : {}".format(len(test_data.labels)))
            test_label_dp = np.reshape(
                test_data.pay_ts - test_data.click_ts > ob_win, (-1, 1))
            test_label = np.reshape(test_data.pay_ts > 0, (-1, 1))
            test_data.labels = np.concatenate(
                [test_label_dp, test_label], axis=1)
        elif "fsiw1" in name:
            cd = parse_float_arg(name, "cd")
            print("cd {}".format(cd))
            train_data = data.sub_days(0, 30).shuffle()
            test_data = data.sub_days(30, 60)
            train_data = train_data.to_fsiw_1(
                cd=cd*SECONDS_A_DAY, T=30*SECONDS_A_DAY)
            test_data = test_data.to_fsiw_1(
                cd=cd*SECONDS_A_DAY, T=60*SECONDS_A_DAY)
        elif "fsiw0" in name:
            cd = parse_float_arg(name, "cd")
            train_data = data.sub_days(0, 30).shuffle()
            test_data = data.sub_days(30, 60)
            train_data = train_data.to_fsiw_0(
                cd=cd*SECONDS_A_DAY, T=30*SECONDS_A_DAY)
            test_data = test_data.to_fsiw_0(
                cd=cd*SECONDS_A_DAY, T=60*SECONDS_A_DAY)
        else:
            raise NotImplementedError("{} dataset does not exist".format(name))
    if params["data_cache_path"] != "None":
        with open(cache_path, "wb") as f:
            pickle.dump({"train": train_data, "test": test_data}, f)
    return {
        "train": {
            "x": train_data.x,
            "click_ts": train_data.click_ts,
            "pay_ts": train_data.pay_ts,
            "sample_ts": train_data.sample_ts,
            "labels": train_data.labels,
            "delay_labels": train_data.delay_labels,
        },
        "test": {
            "x": test_data.x,
            "click_ts": test_data.click_ts,
            "pay_ts": test_data.pay_ts,
            "sample_ts": train_data.sample_ts,
            "labels": test_data.labels,
        }
    }
