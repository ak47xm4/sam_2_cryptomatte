import numpy as np

# pseudocode
# sam 的 png 並沒有前後關係

pngs_np_list = 'imported_pngs'

rand = np.random.randint(0, 5, size=(1080, 1920), dtype=np.uint8)
pngs_np_list_cm_str = "png_file_name  or id"
aaa = sum(pngs_np_list.px.value)

cm_exr = np.image

if (aaa.px.value <= 0):
    exr_0_RGBA = 0
else:
    for o in range(0, len(pngs_np_list)):
        for i in range(0, 5):
            exr_cm_idx = i // 2
            if (rand == i):
                if (i % 2 == 0):
                    cm_exr[exr_cm_idx].R = pngs_np_list_cm_str[o]
                    cm_exr[exr_cm_idx].G = 1
                elif (i % 2 == 1):
                    cm_exr[exr_cm_idx].B = pngs_np_list_cm_str[o]
                    cm_exr[exr_cm_idx].A = 1
                break
