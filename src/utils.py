class ZException(Exception):
    '''
    Exception when z0 is suboptimal.
    '''
    def __init__(self, text, z, *args):
        super(ZException, self).__init__(text, z, *args)
        self.text = text
        self.z = z
