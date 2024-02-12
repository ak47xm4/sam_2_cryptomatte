import OpenEXR

# print(dir(OpenEXR.Header))

exr_file_path = r'C:\Users\Administrator\Documents\GitHub\sam_2_cryptomatte\Scene_ViewLayer_cm.0001.exr'
exr_file = OpenEXR.InputFile(exr_file_path)

header = exr_file.header()

print(header)
