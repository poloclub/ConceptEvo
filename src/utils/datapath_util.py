import os
from datetime import datetime

class DataPathUtil:
    """
    Util functions for data paths
    """

    """
    Constructor
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def gen_sub_directories(self, dir_names):
        """Generate sub-directories for data and log for a given directory name.

        For a given dir_path, it generates 
            self.root_dir
                └── dir_names[0]
                    └── ...
                        └── dir_names[-1]
                            ├── data
                            └── log

        Args:
            - dir_names: directory names

        Returns:
            - data_dir_path: data sub-directory path 
                (self.root_dir/dir_names[0]/.../dir_names[-1]/data)
            - log_dir_path: data sub-directory path 
                (self.root_dir/dir_names[0]/.../dir_names[-1]/log)
        """
        dir_path = self.root_dir
        for dir_name in dir_names:
            dir_path = os.path.join(dir_path, dir_name)
            self.make_dir(dir_path)

        data_dir_path = os.path.join(dir_path, 'data')
        log_dir_path = os.path.join(dir_path, 'log')
        self.make_dir(data_dir_path)
        self.make_dir(log_dir_path)

        return data_dir_path, log_dir_path

    def make_dir(self, dir_path):
        """Generate a directory for a given path, if it does not exist.

        Args:
            - dir_path: a directory path to generate
        
        Returns:
            - N/A
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def is_arg_given(self, arg):
        """Check if a given argument is given by a user.

        Args:
            - arg: an argument
        
        Returns:
            - given_or_not: True if it is given by the user, False otherwise.
        """
        
        arg_type = str(type(arg))
        if arg is None:
            return False
        elif 'bool' in arg_type:
            return arg
        elif 'str' in arg_type:
            if (len(arg) > 0) and (arg.lower() != 'none'):
                return True
            else:
                return False
        elif 'int' in arg_type:
            if arg >= 0:
                return True
            else:
                return False
        elif 'list' in arg_type:
            if len(arg) > 0:
                return True
            else:
                return False
        return True

    def get_time_stamp(self):
        """Get current time stamp.
        
        Args: 
            - N/A

        Returns:
            - time_stamp: a formatted string of current time stamp.
                The format is: "%Y%m%d_%H:%M:%S". i.e., yyyymmdd_hhmmss

        """
        now = datetime.now()
        time_stamp = now.strftime('%Y%m%d_%H:%M:%S')
        return time_stamp