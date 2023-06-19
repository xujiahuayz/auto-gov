import numpy as np

class SumTree(object):
    """
    SumTree is a binary tree data structure where the parent’s value is the sum of its children.
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def add(self, priority, data):
        """
        Add priority and data to the tree.
        """
        tree_idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(tree_idx, priority)
        self.write += 1
        
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, tree_idx, priority):
        """
        Update priority of the tree.
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def total(self):
        """
        Return the root node value.
        """
        return self.tree[0]
    
    def get(self, s):
        """
        Get the leaf_idx, priority and data of the given value.
        """
        parent_idx = 0
        
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            else:
                if s <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                
                else:
                    s -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.capacity + 1
        
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    def __len__(self):
        return self.n_entries
    
    def __repr__(self):
        return f"SumTree(n_entries={self.n_entries}, capacity={self.capacity})"
    
    def __str__(self):
        return self.__repr__()
    
    def __getitem__(self, idx):
        return self.get(idx)
    
    def __setitem__(self, idx, priority):

        self.update(idx, priority)