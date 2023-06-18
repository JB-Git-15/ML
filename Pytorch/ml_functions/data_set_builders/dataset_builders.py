class MnistDataSet(Dataset):
    def __init__(self, dataset_addr):
        self.df        = pd.read_csv(dataset_addr)
        self.df_labels = self.df['label']
        self.df        = self.df.drop(columns=['label'])
        self.dataset   = torch.reshape(torch.tensor(self.df.to_numpy()).float(), (len(self.df), 1, 28, 28))
        self.labels    = torch.tensor(self.df_labels.to_numpy().reshape(-1)).long()
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]
    
    def classes(self):
        return self.labels.unique()
