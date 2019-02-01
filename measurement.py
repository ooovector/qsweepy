'''
Possible types are:
pna-p1D-2D : 1D parameter set with 1 point sweep of PNA -------------- 1
pna-p1D-3D : 1D parameter set with many-point sweep of PNA ----------- 2
pna-p2D-3D : 2D parameter set with 1 point sweep of PNA -------------- 3

Numbers needed for backwards compatibility with Python 3 pickle and because of lack of enum class
'''

class Measurement():

    def __init__(self, type_str="", data=None):
        self.set_type_from_string(type_str)
        self.__data__ = data
    
    def set_type_from_string(self, type_str):
        if type_str == "pna-p1D-2D" or type_str == "p1D-2D":
            self.__type__ = 1
        elif type_str  == "pna-p1D-3D" or type_str == "p1D-2D":
            self.__type__ = 2
        elif type_str  == "pna-p2D-3D" or type_str == "p1D-2D":
            self.__type__ = 3
    
    def get_type_str(self):
        if self.__type__ == 1:
            return "p1D-2D"
        elif self.__type__ == 2:
            return "p1D-3D"
        elif self.__type__ == 3:
            return "p2D-3D"
    
    def get_data(self):
        return self.__data__
    
    def set_data(self, data):
        self.__data__ = data
    
    def get_type(self):
        return self.__type__
    
    def set_type(self, type):
        self.__type__ = type
    
    def copy(self):
        copy = Measurement()
        copy.set_data(self.__data__)
        copy.set_type(self.__type__)
    
        return copy
        
    def remove_background(self, sweep_number, direction):
        '''
        Subtracts a specified sweep from every sweep and returns a measurement with new data
        WARNING Now it works only for NxM plots where N!=M
        
        Parameters: sweep_number : int
                                    Number which specifies the number of parameter
                                    in a parameter list corresponding to the needed sweep
				    direction : string
				                    In what direction the subtracted sweep is recorded, "x" or "y" 
        Returns:   measurement : Measurement
                                    Measurement with no background
        '''
        
        new = self.copy()
		
        if self.__type__ == 2:
            new_data = self.__data__[0], self.__data__[1], self.__data__[2] - self.__data__[2][sweep_number], self.__data__[3] - self.__data__[3][sweep_number]
            new.set_data(new_data)
        elif self.__type__ == 3:
            if direction == "y":		
                new_data = self.__data__[0], self.__data__[1], self.__data__[2], self.__data__[3] - self.__data__[3][sweep_number], self.__data__[4] - self.__data__[4][sweep_number]
                new.set_data(new_data)
            elif direction == "x":
                new_data = self.__data__[0], self.__data__[1], self.__data__[2], (self.__data__[3].T - self.__data__[3][:, sweep_number]).T, (self.__data__[4].T - self.__data__[4][:, sweep_number]).T
                new.set_data(new_data)
        return new