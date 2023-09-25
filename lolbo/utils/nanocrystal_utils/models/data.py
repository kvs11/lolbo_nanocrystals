import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import MinMaxScaler
import numpy as np

from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.data_utils import minmax

class IrOx_Dataset(Dataset):
    def __init__(self, 
            input_array_path, 
            y_values_array_path, 
            graph_embeds_array_path=None, 
            transform_xs=True, 
            transform_ys=True, 
            input_scaler=None,
            target_scaler=None,
            swap_input_axes=True):
        self.input_array = np.load(input_array_path, allow_pickle=True).astype('float32')
        self.y_values_array = np.load(y_values_array_path, allow_pickle=True).astype('float32')

        self.graph_embeds_array = None
        if graph_embeds_array_path is not None:
            self.graph_embeds_array = np.load(graph_embeds_array_path, 
                                          allow_pickle=True).astype('float32')

        # By default, we use Minmax scaler only
        if transform_xs:
            self.input_array, self.input_scaler = minmax(self.input_array)
        elif input_scaler is not None:
            self.input_scaler = input_scaler

        if transform_ys:
            self.target_scaler = MinMaxScaler()
            self.y_values_array = self.target_scaler.fit_transform(self.y_values_array.reshape(-1, 1))
        elif target_scaler is not None:
            self.target_scaler = target_scaler

        if swap_input_axes:
            # Swap axes for the input vectors from (B, H, C) to (B, C, H) -> Eg: (B, 3, 164)
            print ('shape: {0} -> '.format(self.input_array.shape))
            self.input_array = np.swapaxes(self.input_array, 1, 2)
            print ('shape: {0} '.format(self.input_array.shape))
        
    def __len__(self):
        return len(self.input_array)

    def __getitem__(self, idx):
        input_x = self.input_array[idx]
        target_y = self.y_values_array[idx]
        
        if self.graph_embeds_array is None:
            return (input_x, np.array([])), target_y
        else:
            embed_x = self.graph_embeds_array[idx]
            return (input_x, embed_x), target_y


class IrOxDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size,
        train_dataset,
        validation_dataset,
        num_workers=4,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, 
                          batch_size=self.batch_size, shuffle=True) 

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, num_workers=self.num_workers, 
                          batch_size=self.batch_size, shuffle=False)
