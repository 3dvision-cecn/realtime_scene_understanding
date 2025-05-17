

# An object class to represent a detected object
# with a name and a unique identifier
# The object class is used to represent a detected object

class Object:
    def __init__(self, class_name, unique_id):
        self.class_name = class_name
        self.unique_id = unique_id
        self.mask = None
        self.keypoints = None
        self.descriptors = None
        self.bbox = None # bounding boxes of an object
        self.timestamp = None # timestamp of the object



    def __str__(self):
        return f"Object(class_name={self.class_name}, unique_id={self.unique_id}"