class Path(object):
    @staticmethod
    def db_dir(database):
        
        if database == 'Auslan':
            root_dir = 'Data_ori\\Auslan_Dataset'
            output_dir = 'Data_processed'
            return root_dir, output_dir

        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './model/c3d-pretrained.pth'