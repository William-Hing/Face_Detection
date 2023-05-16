# Importing pathlib.Path from Python’s standard library, along with face_recognition, a third-party library. 
from pathlib import Path
import argparse
import face_recognition
import pickle
from collections import Counter
from PIL import Image, ImageDraw

# Defining a constant for the default encoding path. 
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = 'blue'
TEXT_COLOUR = 'white'

# Three calls to .mldir which set exist_ok to True.
Path('training').mkdir(exist_ok  = True)
Path('output').mkdir(exist_ok = True)
Path('validation').mkdir(exist_ok = True)

import sys
sys.argv=['']
del sys

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument(
    "--validate", action="store_true", help="Validate trained model"
)
parser.add_argument(
    "--test", action="store_true", help="Test the model with an unknown image"
)
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)
parser.add_argument(
    "-f", action="store", help="Path to an image with an unknown face"
)
args = parser.parse_args()

# define encode_faces which uses a for loop to go through each directory in the training set and saves the label from each directory to name, 
# then uses the load_image_file function from face_recognition to load each image. 
def encode_faces(
    model: str = 'hog', encodings_location: Path = DEFAULT_ENCODINGS_PATH
    ) -> None: 
    names = []
    encodings = []
    for filepath in Path('training').glob('*/*'):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)
        # uses face_recognition.face_locations() to detect the locations of faces in each image. This function returns a list of four element tuples, 
        # one for each detected face. The four elements per tuple provide the four coordinates of a box that surround the detected face (bounding box).
        face_locations = face_recognition.face_locations(image, model=model)
        # uses face_recognition.face_locations() to generate encodings for the detected faces in an image. encodings is a numeric representation of facial features
        # thats used to match similar faces by their features. 
        face_encodings = face_recognition.face_encodings(image, face_locations)
        for encoding in face_encodings: 
            # adds the names and their encodings to separate lists. 
            names.append(name)
            encodings.append(encoding)
    # creates a dictionary that puts the names and encodings list together and denotes which list is which. Pickle is then use to save the encodings to disk.
    name_encodings = {'names': names, 'encodings': encodings}
    with encodings_location.open(mode = 'wb') as f: 
        pickle.dump(name_encodings, f)
#
encode_faces()

# recognize_faces function will be able to open and load the saved face encodings using pickle and then load the image with the faces to be recognized.
def recognize_faces(
    image_location: str, 
    model: str = 'hog', 
     encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    with encodings_location.open(mode = 'rb') as f: 
        loaded_encodings = pickle.load(f)
    input_image = face_recognition.load_image_file(image_location)
    # these two lines allow the model to detect faces in the input image and their encodings. 
    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )
    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)
    # this will iterate through input_face_locations and input_face_encodings in parallel using zip(). The function _recognize_face(), passes 
    # the encodings  for the unknown and loaded encodings. 
    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name: 
            # added conditional statement which assigns unknown to name if _recognize_face() doesnt find a match. 
            # the name is then printed and the coordinates of the identified faces are saved in bounding_box. 
            name = 'Unknown'
        _display_face(draw, bounding_box, name)
    del draw
    pillow_image.show()
    
    def _recognize_face(unknown_encoding, loaded_encodings): 
        boolean_matches = face_recognition.compare_faces(loaded_encodings['encodings'], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings['names'])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]\
            
def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline = BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbox( 
    (left, bottom), name
    )

def validata(model: str = 'hog'):
    for filepath in Path('validation').rglob('*'):
        if filepath.is_file(): 
            recognize_faces(
                image_location = str(filepath, absolute()), model = model
            )

if __name__ == "__main__":
    if args.train:
        encode_known_faces(model = args.m)
    if args.validate: 
        validate(model = args.m)
    if args.test: 
        recognize_faces(image_location = args.f, model = args.m)