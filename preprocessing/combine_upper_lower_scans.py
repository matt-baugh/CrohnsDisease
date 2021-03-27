from os import listdir
from os.path import isfile, join
import re
import SimpleITK as sitk

data_path = '/vol/bitbucket/mb4617/MRI_Crohns'
folder_paths = [join(data_path, f) for f in ['A', 'I']]

for folder in folder_paths:
    for file_name in listdir(folder):

        upper_match = re.match('(\w\d\d? .*)([Uu][Pp][Pp][Ee][Rr])(.*)$', file_name)

        if upper_match is None:
            continue

        upper_path = join(folder, file_name)
        assert isfile(upper_path), 'File name contains upper but is not file: ' + file_name

        file_name_start, upper_text, file_name_end = upper_match.groups()

        lower_text = 'lower'
        if upper_text.isupper():
            lower_text = lower_text.upper()
        elif upper_text.istitle():
            lower_text = lower_text.capitalize()
        else:
            assert upper_text.islower(), 'Upper text must be uppercase, lowercase or capitalised: ' + upper_text

        lower_name = file_name_start + lower_text + file_name_end
        lower_path = join(folder, lower_name)
        assert isfile(lower_path), f'Lower version of file "{file_name}" does not exist: {lower_name}'

        upper_image = sitk.ReadImage(upper_path)
        lower_image = sitk.ReadImage(lower_path)

        print(file_name)
        print("Size: ", upper_image.GetSize())
        print("Spacing: ", upper_image.GetSpacing())
        print("Origin: ", upper_image.GetOrigin())
        print("Direction: ", upper_image.GetDirection())

        print(lower_name)
        print("Size: ", lower_image.GetSize())
        print("Spacing: ", lower_image.GetSpacing())
        print("Origin: ", lower_image.GetOrigin())
        print("Direction: ", lower_image.GetDirection())
        print()


