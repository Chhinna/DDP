from torch.utils.data import Dataset
import torch

class Memory(Dataset):
    def __init__(self):
        """
        Initialize a Dataset object
        Args:
            self: The Dataset object being initialized
        Returns:
            None: Nothing is returned
        - Initialize the data and labels attributes to None
        - Call the parent class' __init__ method to complete initialization"""
        super(Dataset, self).__init__()
        self.data = None
        self.labels = None
        
#     def additem(data, label):
#         self.data.append(data)
#         self.labels.append(label)
        
    def additems(self, data, label):
        """Adds data and labels to existing data and labels.
        Args:
            data: Data to add
            label: Labels to add
        Returns: 
            None: Does not return anything
        - Checks if self.data already exists, if not initializes it with the input data
        - If self.data exists, concatenates the input data to it along dimension 0
        - Checks if self.labels already exists, if not initializes it with the input labels  
        - If self.labels exists, concatenates the input labels to it along dimension 0"""
        if self.data is None:
            self.data = data
            self.labels = label
        else:
            self.data = torch.cat((self.data, data), dim=0)
            self.labels = torch.cat((self.labels, label), dim=0)
        
    def __getitem__(self, item):
        """
        Returns data and label for given index
        Args:
            item: Index of item to retrieve
        Returns: 
            (data, label): Tuple containing data and label at given index
        Retrieves data and label from internal storage:
            - Fetch data item from self.data using given index
            - Fetch label item from self.labels using same index 
            - Return tuple containing (data, label)
        """
        return (self.data[item], self.labels[item])
    
    def __len__(self):
        """
        Returns the length (number of items) of the object.
        Args:
            self: The object.
        Returns:
            int: The length (number of items) of the object.
        - Check if self.labels is None
        - If None, return 0
        - Else return the length of self.labels using len() function"""
        if self.labels is None:
            return 0
        return len(self.labels)
    