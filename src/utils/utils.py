class TopKKeeper:
    """
    Keep k (key, val, content) paris, where they are sorted in the descending 
    order. If a new inserted pair (key', val', content') has lower value than 
    the minimum value of the item (i.e., val' < min val), it won't be inserted.
    Content is optional.
    """
    
    def __init__(self, k):
        self.k = k
        self.keys = []
        self.vals = []
        self.contents = []


    def will_insert(self, val):
        if len(self.vals) < self.k:
            return True
        elif self.vals[-1] < val:
            return True
        return False

        
    def insert(self, val, key=None, content=None):
        # Check whether we want to insert the item or not
        if not self.will_insert(val):
            return
        
        # Insert
        reach_end = True
        for i, e in enumerate(self.vals):
            if e < val:
                if key is not None:
                    self.keys = self.keys[:i] + [key] + self.keys[i:]
                    self.keys = self.keys[:self.k]

                self.vals = self.vals[:i] + [val] + self.vals[i:]
                self.vals = self.vals[:self.k]

                if content is not None:
                    self.contents = \
                        self.contents[:i] + [content] + self.contents[i:]
                    self.contents = self.contents[:self.k]

                reach_end = False
                break
        if reach_end:
            if key is not None:
                self.keys.append(key)
                self.keys = self.keys[:self.k]

            self.vals.append(val)
            self.vals = self.vals[:self.k]

            if content is not None:
                self.contents.append(content)
                self.contents = self.contents[:self.k]
