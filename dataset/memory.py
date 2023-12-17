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
        - Initialize the parent class
        - Set the data attribute to None initially 
        - Set the labels attribute to None initially
        - Prepare the object to hold data and labels later"""
        super(Dataset, self).__init__()
        self.data = None
        self.labels = None
        
#     def additem(data, label):
#         self.data.append(data)
#         self.labels.append(label)
        
    def additems(self, data, label):
        """Adds data and labels to existing items
        Args:
            data: Data to add
            label: Labels corresponding to data
        Returns: 
            None: Does not return anything
        - Checks if self.data already contains data
        - If empty, initializes self.data and self.labels with input data and labels
        - If not empty, concatenates input data and labels to existing self.data and self.labels along first dimension"""
        if self.data is None:
            self.data = data
            self.labels = label
        else:
            self.data = torch.cat((self.data, data), dim=0)
            self.labels = torch.cat((self.labels, label), dim=0)
        
    def __getitem__(self, item):
        """
        Returns data and label tuple for given index
        Args:
            item: Index to retrieve data and label from
        Returns: 
            tuple: Tuple containing data and label at given index
        Retrieves data and label:
            - Fetch data at given index from self.data
            - Fetch label at given index from self.labels
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
    